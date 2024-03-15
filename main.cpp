/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vkdd.hpp"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include "vk_ddisplay_app.hpp"

int main(uint32_t argc, char const* argv[])
{
  NVPSystem sys(PROJECT_NAME);
  LOGI("Project\t| %s\n", "NVIDIA " PROJECT_NAME);
  LOGI("Version\t| %d.%d\n", VERSION_MAJOR, VERSION_MINOR);
  LOGI("Commit\t| %s%s\n\n", BUILD_STRING, BUILD_UNCOMMITTED_CHANGES ? " (build contains uncommitted changes)" : "");

  VULKAN_HPP_DEFAULT_DISPATCHER.init();
  vk::ApplicationInfo      appInfo("NVIDIA " PROJECT_NAME, 1, "nvpro-samples-engine", 1, VK_API_VERSION_1_3);
  std::vector<char const*> extensions = {"VK_KHR_display", "VK_KHR_surface", "VK_EXT_direct_mode_display"};
#ifndef NDEBUG
  extensions.emplace_back("VK_EXT_debug_utils");
#endif
  vk::InstanceCreateInfo instanceCreateInfo({}, &appInfo, {}, extensions);
  vk::UniqueInstance     instance = vk::createInstanceUnique(instanceCreateInfo);
  VULKAN_HPP_DEFAULT_DISPATCHER.init(instance.get());
  return vkdd::VkDDisplayApp(std::move(instance)).run("NVIDIA " PROJECT_NAME, argc, argv, 1280, 720);
}