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

#version 450
#extension GL_GOOGLE_include_directive : require

#include "perlin.glsl"

vec2 rotate(vec2 src, float angle)
{
  float sina = sin(angle);
  float cosa = cos(angle);
  return vec2(cosa * src.x - sina * src.y, sina * src.x + cosa * src.y);
}

float perlinLayered(vec2 tex)
{
  float v = 0.5f * perlin(18.0f * rotate(tex, -0.2f));
  v += 0.25f * perlin(31.0f * rotate(tex, 0.3f));
  v += 0.125f * perlin(65.0f * rotate(tex, 0.7f));
  v += 0.0625f * perlin(123.0f * rotate(tex, -0.1f));
  return 0.5f + 8.0f / 15.0f * v;
}

vec3 toVec3Color(uint value)
{
  return vec3(float((value >> 16) & 0xff), float((value) >> 8 & 0xff), float(value & 0xff)) / 255.0f;
}

const uint g_numGradients = 6;
const uint g_gradientSize = 3;
vec3 g_gradients[g_numGradients][g_gradientSize] = {{toVec3Color(0xfff1bf), toVec3Color(0xec458d), toVec3Color(0x474ed7)},
                                                    {toVec3Color(0x2c6cbc), toVec3Color(0x71c3f7), toVec3Color(0xf6f6f6)},
                                                    {toVec3Color(0x1a2766), toVec3Color(0xae1b1e), toVec3Color(0xfc9f32)},
                                                    {toVec3Color(0x074170), toVec3Color(0x7e9012), toVec3Color(0xfff708)},
                                                    {toVec3Color(0x40E0D0), toVec3Color(0xFF8C00), toVec3Color(0xFF0080)},
                                                    {toVec3Color(0xff0000), toVec3Color(0x00ff00), toVec3Color(0x0000ff)}};

vec3 sampleColorGradient(uint colorGradientIdx, float t)
{
  t        = clamp(t, 0.0f, 1.0f);
  float dt = 1.0f / (g_gradientSize - 1);
  float t0 = 0.0f;
  uint  i0 = 0;
  while(t0 + dt < t)
  {
    t0 += dt;
    ++i0;
  }
  return mix(g_gradients[colorGradientIdx][i0], g_gradients[colorGradientIdx][i0 + 1], (t - t0) / dt);
}

layout(location = 0) in vec3 fPos;
layout(location = 1) in vec3 fNormal;
layout(location = 2) in vec2 fTex;
layout(location = 3) in flat uint fUniqueId;
layout(location = 4) in float fShellHeight;

layout(location = 0) out vec4 outColor;

void main()
{
  vec2  tex = rotate(fTex, mix(fTex.x, fTex.y, sin(fTex.x + fTex.y)));
  float ao  = 0.6f;
  if(0.01f < fShellHeight)
  {
    vec2  baseTex = 128.0f * vec2(tex.x, 2.0f * tex.y);
    float offset  = distance(floor(baseTex) + 0.5f, baseTex);
    float height  = 0.5f + 0.5f * perlin(32.0f * vec2(tex.x, 2.0f * tex.y));
    float cone    = (height - fShellHeight) / (2.0f * height);
    if(cone < offset)
    {
      discard;
    }
    ao = mix(ao, 1.0f, fShellHeight / height);
  }

  float noise = perlinLayered(vec2(tex.x, 2.0f * tex.y));
  noise       = smoothstep(0.0f, 1.0f, noise);
  vec3 color  = sampleColorGradient(fUniqueId % g_numGradients, noise);
  //color       = vec3(noise);

  vec3 n = normalize(fNormal);

  vec3 skyColor    = vec3(0.9f, 0.9f, 1.0f);
  vec3 groundColor = vec3(0.1f, 0.2f, 0.25f);
  color *= mix(groundColor, skyColor, 0.5f + 0.5f * n.y) * ao;
  outColor = vec4(color, 1.0f);
}