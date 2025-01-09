
#include "PrimitiveTypes.hpp"
#include "UtilFunctions.hpp"
#include "SIMD_Utils.hpp"

#include "CPUInfo.hpp"

auto main() -> int {
	i32 vals[8] = { 5, 4, 3, 2, 1, 0, 50, 40 };
	auto val256 = SIMD::fill256(vals);
	i32 vals2[8] = { 20, 21, 22, 23, 24, 25, -25, 10 };
	auto val256_2 = SIMD::fill256(vals2);

	auto res = _mm256_add_epi32(val256, val256_2);
	auto res2 = SIMD::add<i32>(val256, val256_2);

	SIMD::print256<i32>(val256);
	SIMD::print256<i32>(val256_2);
	SIMD::print256<i32>(res);
	SIMD::print256<i32>(res2);

	return 0;
}