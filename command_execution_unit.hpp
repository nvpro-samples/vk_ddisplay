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
#include "vkdd.hpp"

namespace vkdd {
class CommandExecutionUnit
{
public:
  CommandExecutionUnit(class LogicalDevice& logicalDevice);
  ~CommandExecutionUnit();

  vk::Result        waitForIdle();
  void              waitForIdleAndReset();
  vk::CommandBuffer requestCommandBuffer(uint32_t queueFamilyIndex, std::optional<DeviceMask> deviceMask = {});
  std::vector<vk::CommandBuffer> requestCommandBuffers(std::vector<uint32_t>     queueFamilyIndices,
                                                       std::optional<DeviceMask> deviceMask = {});
  void                           pushWait(vk::CommandBuffer cmdBuffer, vk::SemaphoreSubmitInfo waitSemaphoreInfo);
  void                           pushSignal(vk::CommandBuffer cmdBuffer, vk::SemaphoreSubmitInfo signalSemaphoreInfo);
  void pushWaits(vk::CommandBuffer cmdBuffer, std::vector<vk::SemaphoreSubmitInfo> const& waitSemaphoreInfos);
  void pushSignals(vk::CommandBuffer cmdBuffer, std::vector<vk::SemaphoreSubmitInfo> const& signalSemaphoreInfos);
  void submit();

private:
  LogicalDevice&                                                m_logicalDevice;
  std::mutex                                                    m_mutex;
  std::unordered_map<uint32_t, struct QueueFamilyIndexData>     m_library;
  std::unordered_map<VkCommandBuffer, struct CommandBufferInfo> m_commandBufferInfos;
  std::unordered_map<uint32_t, std::vector<vk::CommandBuffer>>  m_submitOrder;

  vk::CommandBuffer requestCommandBufferUnguarded(uint32_t queueFamilyIndex, std::optional<DeviceMask> deviceMask = {});
};
}  // namespace vkdd
