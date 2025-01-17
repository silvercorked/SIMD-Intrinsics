
#include "PrimitiveTypes.hpp"
#include "UtilFunctions.hpp"
#include "SIMD_Utils.hpp"

#include "CPUInfo.hpp"

#include <ranges>

auto main() -> int {
	i32 vals[8] = { 10, 20, 30, 40, -10, -20, -30, -40 };
	auto val256 = SIMD::fill256(vals);
	i32 vals2[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
	auto val256_2 = SIMD::fill256(vals2);
	std::array<i32, 100> vals3;
	for (auto i = 0; i < 100; i++)
		vals3[i] = i;

	std::vector<__m256i> resFilling;
	for (auto i = 0; i < 100; i += (256 / 32)) {
		if (100 - i < 8) {
			resFilling.push_back(SIMD::fill256<i32>(vals3.data() + i, 100 - i));
		}
		else {
			resFilling.push_back(SIMD::fill256<i32>(vals3.data() + i));
		}
	}

	auto res = _mm256_add_epi32(val256, val256_2);
	auto res2 = SIMD::add<i32>(val256, val256_2);

	auto res3 = _mm256_sub_epi32(val256, val256_2);
	auto res4 = SIMD::subtract<i32>(val256, val256_2);

	auto res5 = _mm256_mul_epi32(val256, val256_2);
	auto res6 = SIMD::multiply<i32>(val256, val256_2);

	auto res7 = _mm256_div_epi32(val256, val256_2);
	auto res8 = SIMD::divide<i32>(val256, val256_2);

	//auto res5 = _mm256_mask_add_epi32(val256, 0x0F, val256, val256_2);
	//auto res6 = SIMD::maskAdd<i32>(val256, val256_2, 0x0F);
	//auto res7 = _mm256_mask_sub_epi32(val256, 0x0F, val256, val256_2);
	//auto res8 = SIMD::maskSubtract<i32>(val256, val256_2, 0x0F);
	// Can't test SIMD::mask__ functions. Don't have AVX 512. Curse you old CPU!!!

	SIMD::print256<i32>(SIMD::fillZero<i32, 256>());
	SIMD::print256<i32>(val256);
	SIMD::print256<i32>(val256_2);
	SIMD::print256<i32>(res);
	SIMD::print256<i32>(res2);
	SIMD::print256<i32>(res3);
	SIMD::print256<i32>(res4);
	SIMD::print256<i32>(res5);
	SIMD::print256<i32>(res6);
	SIMD::print256<i32>(res7);
	SIMD::print256<i32>(res8);
	//SIMD::print256<i32>(res5);
	//SIMD::print256<i32>(res6);
	//SIMD::print256<i32>(res7);
	//SIMD::print256<i32>(res8);
	std::cout << "printing resFilling:\n";
	for (auto i = 0; i < resFilling.size(); i++) {
		SIMD::print256<i32>(resFilling[i]);
	}

	return 0;
}