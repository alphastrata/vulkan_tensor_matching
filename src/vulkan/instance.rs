use ash::{vk, Entry, Instance};
use log::debug;
use crate::error::{Result, TensorMatchingError};
use std::ffi::{CStr, CString};

#[derive(Clone)]
pub struct VulkanInstance {
    pub entry: Entry,
    pub instance: Instance,
    pub debug_utils: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl std::fmt::Debug for VulkanInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanInstance")
            .field("entry", &"Entry")
            .field("instance", &"Instance")
            .field("debug_utils", &"DebugUtils")
            .finish()
    }
}

impl VulkanInstance {
    pub fn new(enable_validation: bool) -> Result<Self> {
        let entry = unsafe { Entry::load() }
            .map_err(|e| TensorMatchingError::VulkanEntryLoadError(e.to_string()))?;

        let app_name = CString::new("Tensorial Template Matching")
            .map_err(|e| TensorMatchingError::VulkanEntryLoadError(e.to_string()))?;
        let engine_name = CString::new("Vulkan Tensor Engine")
            .map_err(|e| TensorMatchingError::VulkanEntryLoadError(e.to_string()))?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name.as_c_str())
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(engine_name.as_c_str())
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::make_api_version(0, 1, 3, 0));

        // Check if we're on macOS and need portability extensions
        #[cfg(target_os = "macos")]
        let extension_names = vec![
            ash::ext::debug_utils::NAME.as_ptr(),
            vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr(),
        ];

        #[cfg(not(target_os = "macos"))]
        let extension_names = vec![
            ash::ext::debug_utils::NAME.as_ptr(),
        ];

        // Only try to enable validation if enabled and layers exist
        let layer_names = if true {
            // Check if the validation layer exists
            let available_layers = unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
            let validation_layer_name = c"VK_LAYER_KHRONOS_validation";
            if available_layers.iter().any(|layer| {
                let layer_name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                layer_name == validation_layer_name
            }) {
                debug!("Vulkan validation layers found and enabled");
                vec![validation_layer_name.as_ptr()]
            } else {
                debug!("Vulkan validation layers not available, proceeding without");
                vec![] // Don't use validation if layer isn't available
            }
        } else {
            vec![]
        };

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);

        if !layer_names.is_empty() {
            create_info = create_info.enabled_layer_names(&layer_names);
        }

        // On macOS, we need to set the enumerate portability bit
        #[cfg(target_os = "macos")]
        {
            create_info = create_info.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }

        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(TensorMatchingError::VulkanError)?;

        let debug_utils = if enable_validation && !layer_names.is_empty() {
            let debug_utils_loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
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

            let debug_messenger = unsafe {
                debug_utils_loader.create_debug_utils_messenger(&debug_create_info, None)
            }.map_err(TensorMatchingError::VulkanError)?;
            Some((debug_utils_loader, debug_messenger))
        } else {
            None
        };

        Ok(Self {
            entry,
            instance,
            debug_utils,
        })
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            if let Some((debug_utils, messenger)) = &self.debug_utils {
                debug_utils.destroy_debug_utils_messenger(*messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = unsafe { *p_callback_data };
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        std::borrow::Cow::from("")
    } else {
        unsafe { CStr::from_ptr(callback_data.p_message_id_name) }.to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        std::borrow::Cow::from("")
    } else {
        unsafe { CStr::from_ptr(callback_data.p_message) }.to_string_lossy()
    };

    log::debug!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity, message_type, message_id_name, message_id_number, message
    );

    vk::FALSE
}