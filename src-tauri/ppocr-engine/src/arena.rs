//! Bounded reusable f32 buffers for one native CPU model.

use std::{
    cell::RefCell,
    collections::BTreeMap,
    fmt,
    mem::size_of,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex, MutexGuard, Weak},
};

const DEFAULT_MAX_CACHED_BYTES: usize = 128 * 1024 * 1024;
const DEFAULT_MAX_BUFFER_BYTES: usize = 32 * 1024 * 1024;
const DEFAULT_MAX_BUFFERS_PER_BUCKET: usize = 4;

thread_local! {
    static ACTIVE_ARENAS: RefCell<Vec<Arc<ArenaInner>>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone)]
pub(crate) struct InferenceArena {
    inner: Arc<ArenaInner>,
}

struct ArenaInner {
    state: Mutex<ArenaState>,
    max_cached_bytes: usize,
    max_buffer_bytes: usize,
    max_buffers_per_bucket: usize,
}

#[derive(Default)]
struct ArenaState {
    buckets: BTreeMap<usize, Vec<Vec<f32>>>,
    cached_bytes: usize,
}

impl Default for InferenceArena {
    fn default() -> Self {
        Self::with_limits(
            DEFAULT_MAX_CACHED_BYTES,
            DEFAULT_MAX_BUFFER_BYTES,
            DEFAULT_MAX_BUFFERS_PER_BUCKET,
        )
    }
}

impl InferenceArena {
    fn with_limits(
        max_cached_bytes: usize,
        max_buffer_bytes: usize,
        max_buffers_per_bucket: usize,
    ) -> Self {
        Self {
            inner: Arc::new(ArenaInner {
                state: Mutex::new(ArenaState::default()),
                max_cached_bytes,
                max_buffer_bytes,
                max_buffers_per_bucket,
            }),
        }
    }

    pub(crate) fn scope<R>(&self, run: impl FnOnce() -> R) -> R {
        ACTIVE_ARENAS.with(|arenas| arenas.borrow_mut().push(self.inner.clone()));
        let _guard = ScopeGuard {
            expected: Arc::as_ptr(&self.inner),
        };
        run()
    }

    #[cfg(test)]
    pub(crate) fn cached_buffers(&self) -> usize {
        self.inner.lock_state().buckets.values().map(Vec::len).sum()
    }

    #[cfg(test)]
    pub(crate) fn cached_bytes(&self) -> usize {
        self.inner.lock_state().cached_bytes
    }
}

struct ScopeGuard {
    expected: *const ArenaInner,
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        ACTIVE_ARENAS.with(|arenas| {
            let active = arenas
                .borrow_mut()
                .pop()
                .expect("inference arena scope stack underflow");
            debug_assert_eq!(Arc::as_ptr(&active), self.expected);
        });
    }
}

impl ArenaInner {
    fn lock_state(&self) -> MutexGuard<'_, ArenaState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn take(self: &Arc<Self>, requested: usize) -> Vec<f32> {
        if requested == 0 {
            return Vec::new();
        }
        let bucket_ceiling = requested.checked_next_power_of_two().unwrap_or(requested);
        let mut state = self.lock_state();
        let capacity = state
            .buckets
            .range(requested..=bucket_ceiling)
            .find_map(|(&capacity, buffers)| (!buffers.is_empty()).then_some(capacity));
        if let Some(capacity) = capacity {
            let buffers = state
                .buckets
                .get_mut(&capacity)
                .expect("selected arena bucket");
            let buffer = buffers.pop().expect("non-empty arena bucket");
            if buffers.is_empty() {
                state.buckets.remove(&capacity);
            }
            state.cached_bytes -= capacity * size_of::<f32>();
            return buffer;
        }
        drop(state);
        Vec::with_capacity(bucket_ceiling)
    }

    fn recycle(&self, buffer: Vec<f32>) {
        let capacity = buffer.capacity();
        let Some(bytes) = capacity.checked_mul(size_of::<f32>()) else {
            return;
        };
        if capacity == 0 || bytes > self.max_buffer_bytes {
            return;
        }
        let mut state = self.lock_state();
        if state.cached_bytes.saturating_add(bytes) > self.max_cached_bytes {
            return;
        }
        let bucket = state.buckets.entry(capacity).or_default();
        if bucket.len() >= self.max_buffers_per_bucket {
            return;
        }
        bucket.push(buffer);
        state.cached_bytes += bytes;
    }
}

fn active_arena() -> Option<Weak<ArenaInner>> {
    ACTIVE_ARENAS.with(|arenas| arenas.borrow().last().map(Arc::downgrade))
}

pub(crate) struct Buffer {
    values: Option<Vec<f32>>,
    arena: Option<Weak<ArenaInner>>,
}

#[derive(Clone)]
pub(crate) struct Handle {
    arena: Option<Weak<ArenaInner>>,
}

impl Handle {
    pub(crate) fn current() -> Self {
        Self {
            arena: active_arena(),
        }
    }

    #[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
    pub(crate) fn zeroed(&self, len: usize) -> Buffer {
        Buffer::zeroed_with(self.arena.clone(), len)
    }

    pub(crate) fn for_overwrite(&self, len: usize) -> Buffer {
        Buffer::for_overwrite_with(self.arena.clone(), len)
    }
}

impl Buffer {
    pub(crate) fn zeroed(len: usize) -> Self {
        Self::zeroed_with(active_arena(), len)
    }

    fn zeroed_with(arena: Option<Weak<ArenaInner>>, len: usize) -> Self {
        let mut values = arena
            .as_ref()
            .and_then(Weak::upgrade)
            .map_or_else(|| Vec::with_capacity(len), |arena| arena.take(len));
        values.resize(len, 0.0);
        values.fill(0.0);
        Self {
            values: Some(values),
            arena,
        }
    }

    /// Returns initialized storage whose previous values are unspecified.
    ///
    /// This avoids clearing a recycled allocation. Callers must overwrite every
    /// element that can be observed before constructing an output tensor.
    pub(crate) fn for_overwrite(len: usize) -> Self {
        Self::for_overwrite_with(active_arena(), len)
    }

    fn for_overwrite_with(arena: Option<Weak<ArenaInner>>, len: usize) -> Self {
        let mut values = arena
            .as_ref()
            .and_then(Weak::upgrade)
            .map_or_else(|| Vec::with_capacity(len), |arena| arena.take(len));
        values.resize(len, 0.0);
        Self {
            values: Some(values),
            arena,
        }
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let arena = active_arena();
        let mut values = arena.as_ref().and_then(Weak::upgrade).map_or_else(
            || Vec::with_capacity(capacity),
            |arena| arena.take(capacity),
        );
        values.clear();
        Self {
            values: Some(values),
            arena,
        }
    }

    fn into_parts(mut self) -> (Vec<f32>, Option<Weak<ArenaInner>>) {
        (
            self.values.take().expect("arena buffer already consumed"),
            self.arena.take(),
        )
    }
}

impl Deref for Buffer {
    type Target = Vec<f32>;

    fn deref(&self) -> &Self::Target {
        self.values.as_ref().expect("arena buffer already consumed")
    }
}

impl DerefMut for Buffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.values.as_mut().expect("arena buffer already consumed")
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let Some(values) = self.values.take() else {
            return;
        };
        if let Some(arena) = self.arena.as_ref().and_then(Weak::upgrade) {
            arena.recycle(values);
        }
    }
}

impl fmt::Debug for Buffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Buffer")
            .field("len", &self.len())
            .field("capacity", &self.capacity())
            .finish()
    }
}

pub(crate) struct F32Storage {
    values: Option<Vec<f32>>,
    arena: Option<Weak<ArenaInner>>,
}

impl F32Storage {
    pub(crate) fn unpooled(values: Vec<f32>) -> Self {
        Self {
            values: Some(values),
            arena: None,
        }
    }

    pub(crate) fn pooled(values: Vec<f32>) -> Self {
        Self {
            values: Some(values),
            arena: active_arena(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.values().len()
    }

    pub(crate) fn values(&self) -> &[f32] {
        self.values
            .as_deref()
            .expect("tensor storage already consumed")
    }

    pub(crate) fn values_mut(&mut self) -> &mut Vec<f32> {
        self.values
            .as_mut()
            .expect("tensor storage already consumed")
    }

    pub(crate) fn into_vec(mut self) -> Vec<f32> {
        self.arena = None;
        self.values.take().expect("tensor storage already consumed")
    }
}

impl From<Buffer> for F32Storage {
    fn from(buffer: Buffer) -> Self {
        let (values, arena) = buffer.into_parts();
        Self {
            values: Some(values),
            arena,
        }
    }
}

impl Deref for F32Storage {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        self.values()
    }
}

impl Drop for F32Storage {
    fn drop(&mut self) {
        let Some(values) = self.values.take() else {
            return;
        };
        if let Some(arena) = self.arena.as_ref().and_then(Weak::upgrade) {
            arena.recycle(values);
        }
    }
}

impl fmt::Debug for F32Storage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("F32Storage")
            .field("len", &self.len())
            .field(
                "capacity",
                &self
                    .values
                    .as_ref()
                    .expect("tensor storage already consumed")
                    .capacity(),
            )
            .field("pooled", &self.arena.is_some())
            .finish()
    }
}

pub(crate) trait IntoF32Storage {
    fn into_f32_storage(self) -> F32Storage;
}

impl IntoF32Storage for Vec<f32> {
    fn into_f32_storage(self) -> F32Storage {
        F32Storage::pooled(self)
    }
}

impl IntoF32Storage for Buffer {
    fn into_f32_storage(self) -> F32Storage {
        self.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn reuses_only_released_buffers() {
        let arena = InferenceArena::with_limits(64 * 1024, 64 * 1024, 4);
        arena.scope(|| {
            let first = Buffer::zeroed(1024);
            let first_address = first.as_ptr();
            let second = Buffer::zeroed(1024);
            let second_address = second.as_ptr();
            assert_ne!(first_address, second.as_ptr());
            drop(second);
            let reused = Buffer::zeroed(1024);
            assert_eq!(reused.as_ptr(), second_address);
        });
    }

    #[test]
    fn overwrite_skips_clear_but_zeroed_resets_recycled_values() {
        let arena = InferenceArena::with_limits(64 * 1024, 64 * 1024, 4);
        arena.scope(|| {
            let mut first = Buffer::zeroed(1024);
            first.fill(7.0);
            let address = first.as_ptr();
            drop(first);

            let overwrite = Buffer::for_overwrite(1024);
            assert_eq!(overwrite.as_ptr(), address);
            assert!(overwrite.iter().all(|&value| value == 7.0));
            drop(overwrite);

            let zeroed = Buffer::zeroed(1024);
            assert_eq!(zeroed.as_ptr(), address);
            assert!(zeroed.iter().all(|&value| value == 0.0));
        });
    }

    #[test]
    fn concurrent_borrows_never_alias() {
        let arena = InferenceArena::with_limits(64 * 1024, 64 * 1024, 4);
        arena.scope(|| drop(Buffer::zeroed(1024)));
        let barrier = Arc::new(Barrier::new(2));
        let first_arena = arena.clone();
        let first_barrier = barrier.clone();
        let first = std::thread::spawn(move || {
            first_arena.scope(|| {
                let buffer = Buffer::zeroed(1024);
                let address = buffer.as_ptr() as usize;
                first_barrier.wait();
                first_barrier.wait();
                address
            })
        });
        let second = arena.scope(|| {
            barrier.wait();
            let buffer = Buffer::zeroed(1024);
            let address = buffer.as_ptr() as usize;
            barrier.wait();
            address
        });
        assert_ne!(first.join().unwrap(), second);
    }

    #[test]
    fn enforces_buffer_and_total_limits() {
        let arena = InferenceArena::with_limits(2048, 2048, 2);
        arena.scope(|| {
            let first = Buffer::zeroed(256);
            let second = Buffer::zeroed(256);
            drop(first);
            drop(second);
            drop(Buffer::zeroed(512));
            drop(Buffer::zeroed(1024));
        });
        assert_eq!(arena.cached_buffers(), 2);
        assert_eq!(arena.cached_bytes(), 2048);
    }
}
