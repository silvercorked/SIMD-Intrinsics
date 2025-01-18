
#include "PrimitiveTypes.hpp"
#include "UtilFunctions.hpp"
#include "SIMD_Ops.hpp"
#include "SIMD_Utils.hpp"

#include "CPUInfo.hpp"

#include <ranges>

#include <random>
#include <numeric>

auto main() -> int {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_int_distribution<> dis(1, 20);

	i32 vals[8] = { 10, 20, 30, 40, -10, -20, -30, -40 };
	auto val256 = SIMD::fill256(vals);
	i32 vals2[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
	auto val256_2 = SIMD::fill256(vals2);
	const auto size = 100;
	std::array<i32, size> vals3;
	for (auto i = 0; i < size; i++)
		vals3[i] = dis(gen);

	std::vector<__m256i> resFilling;
	for (auto i = 0; i < size; i += (256 / 32)) {
		if (size - i < 8) {
			resFilling.push_back(SIMD::fill256<i32>(vals3.data() + i, size - i));
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

	auto vecTestData = std::vector<i32>(vals3.begin(), vals3.end());
	auto operation = SIMD::VecOps(vecTestData);
	operation.map<SIMD::Ops::ADD>(5).map<SIMD::Ops::SUBTRACT>(3);
	auto vecRes = operation.result();
	auto vecRes1 = operation.reduce<SIMD::Ops::ADD>();

	std::cout << "printing result of vec op\n";
	for (auto i = 0; i < vecRes1.size(); i++) {
		std::cout << "" << i << ": " << vecRes1[i] << "\n";
	}

	std::transform(vecTestData.begin(), vecTestData.end(), vecTestData.begin(), [](const int& val) { return val + 5 - 3; });
	const auto biggerSize = vecRes.size() > vecTestData.size() ? vecRes.size() : vecTestData.size();
	for (auto i = 0; i < biggerSize; i++) {
		if (i < vecRes.size() && i < vecTestData.size()) {
			std::cout << "" << i << ": " << vecTestData.at(i) << " vs " << vecRes.at(i) << "\n";
		}
		else if (i < vecRes.size()) {
			std::cout << "" << i << ": " << "_" << " vs " << vecRes.at(i) << "\n";
		}
		else if (i < vecTestData.size()) {
			std::cout << "" << i << ": " << vecTestData.at(i) << " vs " << "_" << "\n";
		}
	}
	int result = std::accumulate(vecTestData.begin(), vecTestData.end(), 0, [](int a, int b) { return a + b; });
	std::cout << "result from non-simd ops: " << result << "\n";

	return 0;
}