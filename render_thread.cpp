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

#include "render_thread.hpp"

#include "logical_device.hpp"

namespace vkdd {
RenderThread::RenderThread(class LogicalDevice& logicalDevice, DeviceIndex deviceIndex)
    : m_status(Status::CREATED)
    , m_logicalDevice(logicalDevice)
    , m_deviceIndex(deviceIndex)
    , m_systemPhysicalDeviceIndex((uint32_t)-1)
{
  std::vector<vk::PhysicalDevice> devices = m_logicalDevice.vkInstance().enumeratePhysicalDevices();
  m_systemPhysicalDeviceIndex =
      std::find(devices.begin(), devices.end(), m_logicalDevice.getPhysicalDevice(deviceIndex)) - devices.begin();
}

void RenderThread::start()
{
  m_imageAcquiredSem = m_logicalDevice.vkDevice().createSemaphoreUnique({});
  m_renderDoneSem    = m_logicalDevice.vkDevice().createSemaphoreUnique({});
  m_thread           = std::make_unique<std::thread>([this]() {
    std::unique_lock lock(m_mtx);
    while(m_status != Status::INTERRUPTED)
    {
      if(m_status == Status::RECORDING)
      {
        this->recordCommands(*m_currentCmdExecUnit, m_currentFramebuffer);
      }
      m_status = Status::WAITING;
      m_cv.notify_all();
      m_cv.wait(lock, [this]() { return m_status != Status::WAITING; });
    }
  });
}

void RenderThread::recordCommandsAsync(class CommandExecutionUnit& cmdExecUnit, vk::Framebuffer framebuffer)
{
  std::unique_lock lock(m_mtx);
  m_currentCmdExecUnit = &cmdExecUnit;
  m_currentFramebuffer = framebuffer;
  m_status             = Status::RECORDING;
  m_cv.notify_all();
}

void RenderThread::finishCommandRecording()
{
  std::unique_lock lock(m_mtx);
  if(m_status == Status::RECORDING)
  {
    m_cv.wait(lock, [this]() { return m_status == Status::WAITING; });
  }
}

void RenderThread::interrupt()
{
  std::unique_lock lock(m_mtx);
  m_status = Status::INTERRUPTED;
  m_cv.notify_all();
}

void RenderThread::join()
{
  m_thread->join();
}
}  // namespace vkdd
