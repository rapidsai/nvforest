/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nvforest/detail/padding.hpp>

#include <gtest/gtest.h>

#include <cstdint>

namespace nvforest::detail {

TEST(Padding, OddPaddedSizeKeepsOddValues)
{
  EXPECT_EQ(odd_padded_size(1u), 1u);
  EXPECT_EQ(odd_padded_size(17u), 17u);
  EXPECT_EQ(odd_padded_size(std::uint64_t{4095}), std::uint64_t{4095});
}

TEST(Padding, OddPaddedSizeRoundsEvenValuesUp)
{
  EXPECT_EQ(odd_padded_size(0u), 1u);
  EXPECT_EQ(odd_padded_size(16u), 17u);
  EXPECT_EQ(odd_padded_size(std::uint64_t{4096}), std::uint64_t{4097});
}

}  // namespace nvforest::detail
