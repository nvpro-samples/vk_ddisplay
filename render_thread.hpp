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

#include <functional>
#include <thread>

namespace vkdd {
class RenderThread
{
public:
  RenderThread(class LogicalDevice& logicalDevice, DeviceIndex deviceIndex);

  void start();
  void recordCommandsAsync(class CommandExecutionUnit& cmdExecUnit, vk::Framebuffer framebuffer);
  void finishCommandRecording();
  void interrupt();
  void join();

  virtual void   recordCommands(class CommandExecutionUnit& cmdExecUnit, vk::Framebuffer framebuffer) = 0;
  vk::Semaphore  getImageAcquiredSemaphore() const { return m_renderDoneSem.get(); }
  vk::Semaphore  getRenderDoneSemaphore() const { return m_renderDoneSem.get(); }
  LogicalDevice& getLogicalDevice() const { return m_logicalDevice; }
  DeviceIndex    getDeviceIndex() const { return m_deviceIndex; }
  uint32_t       getSystemPhysicalDeviceIndex() const { return m_systemPhysicalDeviceIndex; }

private:
  enum class Status
  {
    CREATED,
    RECORDING,
    WAITING,
    INTERRUPTED
  };

  Status                       m_status;
  LogicalDevice&               m_logicalDevice;
  DeviceIndex                  m_deviceIndex;
  uint32_t                     m_systemPhysicalDeviceIndex;
  CommandExecutionUnit*        m_currentCmdExecUnit;
  vk::Framebuffer              m_currentFramebuffer;
  vk::UniqueSemaphore          m_imageAcquiredSem;
  vk::UniqueSemaphore          m_renderDoneSem;
  std::unique_ptr<std::thread> m_thread;
  std::mutex                   m_mtx;
  std::condition_variable      m_cv;
};
}  // namespace vkdd
