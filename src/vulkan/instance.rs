use crate::error::Result;
use ash::{Entry, Instance, vk};
use log::debug;
use std::ffi::{CStr, CString};
use std::sync::Arc;

pub struct VulkanInstance {
    pub entry: Entry,
    pub instance: Instance,
    pub debug_utils: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl VulkanInstance {
    pub fn new(enable_validation: bool) -> Result<Arc<Self>> {
        let entry = unsafe { Entry::load()? };

        let app_name = CString::new("Vulkan Tensor Matching").unwrap();
        let engine_name = CString::new("No Engine").unwrap();

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_3);

        let mut extension_names = Vec::new();
        if enable_validation {
            extension_names.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let layer_names = if enable_validation {
            vec![c"VK_LAYER_KHRONOS_validation".as_ptr()]
        } else {
            Vec::new()
        };

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names);

        let instance = unsafe { entry.create_instance(&create_info, None)? };

        let debug_utils = if enable_validation {
            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));

            let debug_utils_loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let debug_messenger =
                unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_info, None)? };
            Some((debug_utils_loader, debug_messenger))
        } else {
            None
        };

        Ok(Arc::new(Self {
            entry,
            instance,
            debug_utils,
        }))
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            debug!("Destroying Vulkan instance");
            if let Some((debug_utils_loader, debug_messenger)) = self.debug_utils.take() {
                debug_utils_loader.destroy_debug_utils_messenger(debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let callback_data = unsafe { *callback_data };
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        "".into()
    } else {
        unsafe { CStr::from_ptr(callback_data.p_message_id_name) }.to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        "".into()
    } else {
        unsafe { CStr::from_ptr(callback_data.p_message) }.to_string_lossy()
    };

    log::debug!(
        "{:?}:\\n{:?} [{} ({})] : {}\\n",
        message_severity,
        message_type,
        message_id_name,
        message_id_number,
        message
    );

    vk::FALSE
}
