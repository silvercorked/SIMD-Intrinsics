#pragma once

#include "PrimitiveTypes.hpp"

#include "PrimitiveConcepts.hpp"
#include "SIMD_Utils.hpp"

#include <stdexcept>
#include <type_traits>
#include <concepts>
#include <tmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

#include <vector>



// In is i32 or u32 and out is __m256i
// or In is f32 and out is __m256
template <typename In, typename Out>
concept In32Out256 = ((std::same_as<In, u32> || std::same_as<In, i32>) && std::same_as<Out, __m256i>)
|| (std::same_as<In, f32> && std::same_as<Out, __m256>);

template <typename In, typename Out> requires In32Out256<In, Out>
auto simd_mapSumSingle(
	In a, In b, In c, In d, In e, In f, In g, In h,
	In toApply
) -> Out {
	auto one = fill32_256(a, b, c, d, e, f, g, h);
	auto two = fill32_256(toApply);
	if constexpr (std::same_as<f32, In>::value) {
		return _mm256_add_ps(one, two);
	}
	else if constexpr (std::same_as<f32, In>::value) {
		return _mm256_add_epi32(one, two);
	}
	else {
		std::runtime_error("Invalid input type");
	}
}

template <Numeric32 T>
class VecOps {
	using BigType = std::conditional_t<Integral<T>, __m256i, __m256>;

	std::vector<T> data;
	BigType res;
	bool hasReduced;
public:
	enum struct Ops: int {
		ADD, // add
		SUBTRACT, // subtract
		MULTIPLY, // multiply
		DIVIDE // divide
	};

	VecOps(const std::vector<T>& data) : data(data), res(0), hasReduced(false) {}

	auto map(Ops op, T nVal) {
	
		return *this; // allows chaining
	}

	auto reduce(Ops op) -> BigType {
		if (this->hasReduced) // can't reduce after reducing once already
			return __m256(0);
		this->hasReduced = true;

		switch (op) {
			case Ops::ADD:
			
				break;
			case Ops::SUBTRACT:
			
				break;
			case Ops::MULTIPLY:

				break;
			case Ops::DIVIDE:

				break;
			case Ops::INTEGRAL_DIVIDE:

				break;
			default:
				std::runtime_error("Unsupported Operation Requested");
		}

		return this->result();
	}

	auto result() {
		if (this->hasReduced) {
			
		}
	}
};
