use std::{
    ffi::c_void,
    mem, ptr,
    sync::{Arc, OnceLock},
};

type Handle = *mut c_void;
type GetSystemCpuSetInformationFn = unsafe extern "system" fn(
    information: *mut u8,
    buffer_length: u32,
    returned_length: *mut u32,
    process: Handle,
    flags: u32,
) -> i32;
type SetThreadSelectedCpuSetsFn = unsafe extern "system" fn(
    thread: Handle,
    cpu_set_ids: *const u32,
    cpu_set_id_count: u32,
) -> i32;

#[link(name = "kernel32")]
unsafe extern "system" {
    #[link_name = "GetModuleHandleW"]
    fn get_module_handle_w(module_name: *const u16) -> Handle;
    #[link_name = "GetProcAddress"]
    fn get_proc_address(module: Handle, procedure_name: *const u8) -> *mut c_void;
    #[link_name = "GetCurrentProcess"]
    fn get_current_process() -> Handle;
    #[link_name = "GetCurrentThread"]
    fn get_current_thread() -> Handle;
}

#[derive(Clone, Copy)]
struct CpuSetApi {
    get_system_information: GetSystemCpuSetInformationFn,
    set_thread_selection: SetThreadSelectedCpuSetsFn,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CpuSet {
    id: u32,
    efficiency_class: u8,
    flags: u8,
}

const CPU_SET_INFORMATION_TYPE: u32 = 0;
const CPU_SET_INFORMATION_SIZE: usize = 32;
const CPU_SET_PARKED: u8 = 0x1;
const CPU_SET_ALLOCATED: u8 = 0x2;
const CPU_SET_ALLOCATED_TO_TARGET_PROCESS: u8 = 0x4;

pub(super) fn preferred_performance_cpu_sets() -> Arc<[u32]> {
    query_cpu_sets()
        .and_then(|buffer| parse_cpu_sets(&buffer))
        .map_or_else(empty_cpu_sets, |cpu_sets| {
            select_performance_cpu_sets(&cpu_sets)
        })
}

pub(super) fn configure_thread(cpu_sets: &[u32]) {
    if cpu_sets.is_empty() {
        return;
    }
    let Some(api) = cpu_set_api() else {
        return;
    };
    let Ok(count) = u32::try_from(cpu_sets.len()) else {
        return;
    };
    // CPU Set selection is soft affinity and lasts only for this dedicated worker.
    unsafe {
        (api.set_thread_selection)(get_current_thread(), cpu_sets.as_ptr(), count);
    }
}

fn empty_cpu_sets() -> Arc<[u32]> {
    Arc::from(Vec::<u32>::new())
}

fn cpu_set_api() -> Option<&'static CpuSetApi> {
    static API: OnceLock<Option<CpuSetApi>> = OnceLock::new();
    API.get_or_init(resolve_cpu_set_api).as_ref()
}

fn resolve_cpu_set_api() -> Option<CpuSetApi> {
    const KERNEL32: [u16; 13] = [
        b'k' as u16,
        b'e' as u16,
        b'r' as u16,
        b'n' as u16,
        b'e' as u16,
        b'l' as u16,
        b'3' as u16,
        b'2' as u16,
        b'.' as u16,
        b'd' as u16,
        b'l' as u16,
        b'l' as u16,
        0,
    ];
    let module = unsafe { get_module_handle_w(KERNEL32.as_ptr()) };
    if module.is_null() {
        return None;
    }
    let get_system_information =
        unsafe { get_proc_address(module, c"GetSystemCpuSetInformation".as_ptr().cast()) };
    let set_thread_selection =
        unsafe { get_proc_address(module, c"SetThreadSelectedCpuSets".as_ptr().cast()) };
    if get_system_information.is_null() || set_thread_selection.is_null() {
        return None;
    }
    Some(CpuSetApi {
        // Function pointers returned by GetProcAddress have the declared Win32 ABI.
        get_system_information: unsafe {
            mem::transmute::<*mut c_void, GetSystemCpuSetInformationFn>(get_system_information)
        },
        set_thread_selection: unsafe {
            mem::transmute::<*mut c_void, SetThreadSelectedCpuSetsFn>(set_thread_selection)
        },
    })
}

fn query_cpu_sets() -> Option<Vec<u8>> {
    let api = cpu_set_api()?;
    let process = unsafe { get_current_process() };
    let mut required = 0u32;
    unsafe {
        (api.get_system_information)(ptr::null_mut(), 0, &mut required, process, 0);
    }
    if required == 0 {
        return None;
    }

    // The CPU Set list can grow between sizing and retrieval, so retry a short race.
    for _ in 0..3 {
        let mut buffer = vec![0u8; required as usize];
        let mut returned = required;
        let succeeded = unsafe {
            (api.get_system_information)(buffer.as_mut_ptr(), required, &mut returned, process, 0)
        } != 0;
        if succeeded {
            if returned as usize > buffer.len() {
                return None;
            }
            buffer.truncate(returned as usize);
            return Some(buffer);
        }
        if returned <= required {
            return None;
        }
        required = returned;
    }
    None
}

fn parse_cpu_sets(buffer: &[u8]) -> Option<Vec<CpuSet>> {
    let mut cpu_sets = Vec::new();
    let mut offset = 0usize;
    while offset < buffer.len() {
        let header = buffer.get(offset..offset.checked_add(8)?)?;
        let size = u32::from_le_bytes(header[..4].try_into().ok()?) as usize;
        let information_type = u32::from_le_bytes(header[4..8].try_into().ok()?);
        if size < 8 {
            return None;
        }
        let end = offset.checked_add(size)?;
        let record = buffer.get(offset..end)?;
        if information_type == CPU_SET_INFORMATION_TYPE {
            if size < CPU_SET_INFORMATION_SIZE {
                return None;
            }
            cpu_sets.push(CpuSet {
                id: u32::from_le_bytes(record[8..12].try_into().ok()?),
                efficiency_class: record[18],
                flags: record[19],
            });
        }
        offset = end;
    }
    Some(cpu_sets)
}

fn select_performance_cpu_sets(cpu_sets: &[CpuSet]) -> Arc<[u32]> {
    let available = cpu_sets.iter().filter(|cpu_set| {
        cpu_set.flags & CPU_SET_PARKED == 0
            && (cpu_set.flags & CPU_SET_ALLOCATED == 0
                || cpu_set.flags & CPU_SET_ALLOCATED_TO_TARGET_PROCESS != 0)
    });
    let Some(minimum_class) = available
        .clone()
        .map(|cpu_set| cpu_set.efficiency_class)
        .min()
    else {
        return empty_cpu_sets();
    };
    let maximum_class = available
        .clone()
        .map(|cpu_set| cpu_set.efficiency_class)
        .max()
        .expect("nonempty CPU Set iterator");
    if minimum_class == maximum_class {
        return empty_cpu_sets();
    }
    Arc::from(
        available
            .filter(|cpu_set| cpu_set.efficiency_class == maximum_class)
            .map(|cpu_set| cpu_set.id)
            .collect::<Vec<_>>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_set_record(id: u32, efficiency_class: u8, flags: u8) -> Vec<u8> {
        let mut record = vec![0u8; CPU_SET_INFORMATION_SIZE];
        record[0..4].copy_from_slice(&(CPU_SET_INFORMATION_SIZE as u32).to_le_bytes());
        record[4..8].copy_from_slice(&CPU_SET_INFORMATION_TYPE.to_le_bytes());
        record[8..12].copy_from_slice(&id.to_le_bytes());
        record[18] = efficiency_class;
        record[19] = flags;
        record
    }

    #[test]
    fn selects_all_available_cpu_sets_in_highest_heterogeneous_class() {
        let mut buffer = cpu_set_record(3, 0, 0);
        buffer.extend(cpu_set_record(17, 8, 0));
        buffer.extend(cpu_set_record(19, 8, 0));

        let cpu_sets = parse_cpu_sets(&buffer).expect("valid records");
        assert_eq!(&*select_performance_cpu_sets(&cpu_sets), &[17, 19]);
    }

    #[test]
    fn filters_parked_and_unavailable_exclusive_cpu_sets() {
        let mut buffer = cpu_set_record(1, 0, 0);
        buffer.extend(cpu_set_record(2, 9, CPU_SET_PARKED));
        buffer.extend(cpu_set_record(3, 9, CPU_SET_ALLOCATED));
        buffer.extend(cpu_set_record(
            4,
            8,
            CPU_SET_ALLOCATED | CPU_SET_ALLOCATED_TO_TARGET_PROCESS,
        ));

        let cpu_sets = parse_cpu_sets(&buffer).expect("valid records");
        assert_eq!(&*select_performance_cpu_sets(&cpu_sets), &[4]);
    }

    #[test]
    fn homogeneous_cpu_sets_leave_windows_scheduling_unchanged() {
        let mut buffer = cpu_set_record(7, 4, 0);
        buffer.extend(cpu_set_record(11, 4, 0));

        let cpu_sets = parse_cpu_sets(&buffer).expect("valid records");
        assert!(select_performance_cpu_sets(&cpu_sets).is_empty());
    }

    #[test]
    fn rejects_truncated_and_invalid_record_sizes() {
        let mut truncated = cpu_set_record(1, 0, 0);
        truncated.pop();
        assert!(parse_cpu_sets(&truncated).is_none());

        let mut too_small = cpu_set_record(1, 0, 0);
        too_small[0..4].copy_from_slice(&31u32.to_le_bytes());
        assert!(parse_cpu_sets(&too_small[..31]).is_none());

        let mut too_large = cpu_set_record(1, 0, 0);
        too_large[0..4].copy_from_slice(&64u32.to_le_bytes());
        assert!(parse_cpu_sets(&too_large).is_none());

        let mut zero = cpu_set_record(1, 0, 0);
        zero[0..4].copy_from_slice(&0u32.to_le_bytes());
        assert!(parse_cpu_sets(&zero).is_none());
    }
}
