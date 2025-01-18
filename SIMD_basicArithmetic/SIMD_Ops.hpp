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

namespace SIMD {
	enum struct Ops: int {
		ADD, // add
		SUBTRACT, // subtract
		MULTIPLY, // multiply
		DIVIDE // divide
	};

	// due to lack of support for more types than these in a number of spaces, ill just use these for the moment.
	template <typename T>
	concept I32_or_F32 = std::same_as<T, i32> || std::same_as<T, f32>;

	template <I32_or_F32 T>
	class VecOps {
		using BigType = std::conditional_t<
			std::same_as<T, i32>, __m256i, std::conditional_t<
			std::same_as<T, f32>, __m256, std::false_type>
		>;

		std::vector<BigType> data;
		const size_t originalSize;
		const size_t amountInBigType;
		bool hasReduced;
	public:
		VecOps(const std::vector<T>& input) :
			originalSize(input.size()),
			amountInBigType(sizeof(BigType) / sizeof(T)),
			hasReduced(false)
		{
			if (this->originalSize == 0) {
				throw std::runtime_error("input vector must have data");
			}
			const auto dataSize = this->originalSize / this->amountInBigType;
			const auto remainder = input.size() % this->amountInBigType;
			if (remainder != 0)
				this->data.reserve(dataSize + 1);
			else
				this->data.reserve(dataSize);

			for (auto i = 0; i < this->originalSize; i += this->amountInBigType) {
				if (this->originalSize - i < this->amountInBigType) {
					this->data.push_back(SIMD::fill256<T>(input.data() + i, this->originalSize - i));
				}
				else {
					this->data.push_back(SIMD::fill256<T>(input.data() + i));
				}
			}
		}

		template <Ops OP>
		auto map(T nVal) -> VecOps<T>& {
			if (OP == Ops::DIVIDE && nVal == 0) {
				throw std::runtime_error("illegal argument. Cannot divide by zero!");
			}
			const auto valVec = SIMD::fill256<T>(nVal);
			for (auto i = 0; i < this->data.size(); i++) {
				this->data[i] = this->applyOPSIMD<OP>(this->data[i], valVec);
			}
			return *this; // allows chaining
		}

		// only supporting additive and multiplicative reductions since order doesn't matter
		template <Ops OP> requires (OP == Ops::ADD || OP == Ops::MULTIPLY)
		auto reduce() {
			if (this->hasReduced)
				return this->result(); // avoid consequences of recomputing and just give result
			this->hasReduced = true;
			auto currSize = this->data.size();
			if (currSize > 1) {
				if (size_t remainder = this->originalSize % this->amountInBigType; remainder != 0) {
					// add & subtract can start filling extra values. These are ignored when giving data back before reduction
					// but will affect result unless they're dealt with
					T* asTs = pun_cast<T*>(&(this->data[currSize - 1]));
					for (auto i = remainder; i < this->amountInBigType; i++) {
						if constexpr (OP == Ops::ADD) {
							asTs[i] = 0; // won't affect anything if added or subtracted
						}
						else if constexpr (OP == Ops::MULTIPLY) {
							asTs[i] = 1; // won't affect anything if multiplied or divided
						}
						else {
							throw std::runtime_error("invalid operation");
						}
					}
				}
				size_t i = 0;
				while (currSize > 1) {
					if (currSize & 1) { // check for odd and get to even for simpler reduction
						this->data[currSize - 2] = this->applyOPSIMD<OP>(this->data[currSize - 2], this->data[currSize - 1]);
						currSize--;
					}
					while (true) {
						this->data[i] = this->applyOPSIMD<OP>(this->data[i * 2], this->data[(i * 2) + 1]);
						i++;
						if (i * 2 >= currSize) {
							i = 0;
							currSize /= 2;
							break;
						}
					}
				}
			}

			currSize = this->amountInBigType; // reuse
			T* asTs = pun_cast<T*>(&(this->data[0])); // guarenteed this->amountInBigType is even
			size_t i = 0;
			while (currSize > 1) {
				asTs[i] = this->applyOP<OP>(asTs[i * 2], asTs[(i * 2) + 1]);
				i++;
				if (i * 2 >= currSize) {
					i = 0;
					currSize /= 2;
				}
			}
			return this->result();
		}

		auto result() -> std::vector<T> {
			if (this->hasReduced) {
				return std::vector<T>(1,
					pun_cast<T*>(
						&(this->data[0]) // get bigtype pointer, then cast to T*
					)[0] // then get first element
				); // and use it to form vector w/ 1 element
			} // in reductions, final result will be stored in the first element of a bigType in the data vector
			// otherwise, convert from bigType array to T array and return
			size_t remainder = this->originalSize % this->amountInBigType;
			std::vector<T> toReturn; // held onto original size for this
			toReturn.reserve(this->originalSize);
			for (auto i = 0; i < this->data.size(); i++) {
				T* asTs = pun_cast<T*>(&(this->data[i]));
				for (
					auto j = 0;
					j < (i == this->data.size() - 1 ? remainder : this->amountInBigType);
					j++
				) {
					toReturn.push_back(asTs[j]);
				}
			}
			return toReturn;
		}

	private:
		template <Ops OP>
		auto applyOPSIMD(const BigType& a, const BigType& b) -> BigType {
			if constexpr (OP == Ops::ADD) {
				return SIMD::add<T>(a, b);
			}
			else if constexpr (OP == Ops::SUBTRACT) {
				return SIMD::subtract<T>(a, b);
			}
			else if constexpr (OP == Ops::MULTIPLY) {
				return SIMD::multiply<T>(a, b);
			}
			else if constexpr (OP == Ops::DIVIDE) {
				return SIMD::divide<T>(a, b);
			}
			else {
				throw std::runtime_error("Unsupported Operation Requested");
			}
		}
		template <Ops OP>
		auto applyOP(const T a, const T b) -> T {
			if constexpr (OP == Ops::ADD) {
				return a + b;
			}
			else if constexpr (OP == Ops::SUBTRACT) {
				return a - b;
			}
			else if constexpr (OP == Ops::MULTIPLY) {
				return a - b;
			}
			else if constexpr (OP == Ops::DIVIDE) {
				return a / b;
			}
			else {
				throw std::runtime_error("Unsupported Operation Requested");
			}
		}
	};

};
