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

#ifndef PERLIN_GLSL
#define PERLIN_GLSL

vec2 randomGradient(int ix, int iy)
{
  const uint w = 32;
  const uint s = w / 2;  // rotation width
  uint       a = ix, b = iy;
  a *= 3284157443;
  b ^= a << s | a >> w - s;
  b *= 1911520717;
  a ^= b << s | b >> w - s;
  a *= 2048419325;
  float random = a * (3.14159265 / ~(~0u >> 1));
  vec2  v;
  v.x = cos(random);
  v.y = sin(random);
  return v;
}

float dotGridGradient(int ix, int iy, vec2 coords)
{
  vec2 gradient = randomGradient(ix, iy);
  vec2 offset   = coords - vec2(float(ix), float(iy));
  return dot(gradient, offset);
}

float perlin(vec2 coords)
{
  int x0 = int(floor(coords.x));
  int x1 = x0 + 1;
  int y0 = int(floor(coords.y));
  int y1 = y0 + 1;

  float sx = coords.x - float(x0);
  float sy = coords.y - float(y0);

  float n0  = dotGridGradient(x0, y0, coords);
  float n1  = dotGridGradient(x1, y0, coords);
  float n2  = dotGridGradient(x0, y1, coords);
  float n3  = dotGridGradient(x1, y1, coords);
  float ix0 = mix(n0, n1, sx);
  float ix1 = mix(n2, n3, sx);
  return mix(ix0, ix1, sy);
}

#endif