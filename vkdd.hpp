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

#pragma once
#include "math_util.hpp"
#include "version.h"

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include "vulkan/vulkan.hpp"

#include "nvh/nvprint.hpp"

#include <iostream>
#include <mutex>
#include <optional>

#define VKDD_TODO(_msg)                                                                                                \
  do                                                                                                                   \
  {                                                                                                                    \
    std::cerr << "@todo: " << _msg << std::endl;                                                                       \
    assert(false);                                                                                                     \
  } while(0)

namespace vkdd {
struct Colors
{
  inline static Vec3f const RED        = {1.0f, 0.0f, 0.0f};
  inline static Vec3f const GREEN      = {0.0f, 1.0f, 0.0f};
  inline static Vec3f const BLUE       = {0.0f, 0.0f, 1.0f};
  inline static Vec3f const CYAN       = {0.0f, 1.0f, 1.0f};
  inline static Vec3f const MAGENTA    = {1.0f, 0.0f, 1.0f};
  inline static Vec3f const YELLOW     = {1.0f, 1.0f, 0.0f};
  inline static Vec3f const DARK_GRAY  = {0.25f, 0.25f, 0.25f};
  inline static Vec3f const GRAY       = {0.5f, 0.5f, 0.5f};
  inline static Vec3f const LIGHT_GRAY = {0.75f, 0.75f, 0.75f};
  inline static Vec3f const BLACK      = {0.0f, 0.0f, 0.0f};
  inline static Vec3f const WHITE      = {1.0f, 1.0f, 1.0f};
  inline static Vec3f const STRONG_RED = {0.725f, 0.471f, 0.0f};
  inline static Vec3f const GREEN_NV   = {0.462f, 0.725f, 0.0f};
  inline static Vec3f const BONDI_BLUE = {0.0f, 0.588f, 0.725f};
};

const uint32_t NUM_QUEUED_FRAMES = 4;

typedef uint64_t                   FrameIndex;
typedef uint32_t                   DeviceIndex;
typedef std::optional<DeviceIndex> OptionalDeviceIndex;

struct DeviceMask
{
public:
  static DeviceMask ofSingleDevice(DeviceIndex deviceIndex)
  {
    DeviceMask deviceMask;
    deviceMask.add(deviceIndex);
    return deviceMask;
  }

  void add(DeviceIndex deviceIndex) { m_bits |= 1 << (uint32_t)deviceIndex; }
       operator uint32_t() const { return m_bits; }

private:
  uint32_t m_bits = 0;
};

static std::string formatVkDeviceName(vk::PhysicalDevice device)
{
  vk::PhysicalDeviceIDProperties idProps;
  vk::PhysicalDeviceProperties2  props;
  props.pNext = &idProps;
  device.getProperties2(&props);
  char buffer[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE + 43];
  sprintf_s(buffer, "%s {%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x}",
            props.properties.deviceName.data(), idProps.deviceUUID[15], idProps.deviceUUID[14], idProps.deviceUUID[13],
            idProps.deviceUUID[12], idProps.deviceUUID[11], idProps.deviceUUID[10], idProps.deviceUUID[9],
            idProps.deviceUUID[8], idProps.deviceUUID[7], idProps.deviceUUID[6], idProps.deviceUUID[5], idProps.deviceUUID[4],
            idProps.deviceUUID[3], idProps.deviceUUID[2], idProps.deviceUUID[1], idProps.deviceUUID[0]);
  return buffer;
}
}  // namespace vkdd