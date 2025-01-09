#pragma once

#include "PrimitiveTypes.hpp"

#include "PrimitiveConcepts.hpp"

#include <stdexcept>
#include <concepts>
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#include <iostream>
#include <format>
#include <string>

namespace SIMD {
	template <typename In, typename Out>
	concept SIMDInOut128_Type = (std::same_as<Out, __m128i>
		&& (
			(std::same_as<In, u8> || std::same_as<In, i8>)
			|| (std::same_as<In, u16> || std::same_as<In, i16>)
			|| (std::same_as<In, u32> || std::same_as<In, i32>)
			|| (std::same_as<In, u64> || std::same_as<In, i64>)
			)
		)
		|| (std::same_as<Out, __m128> && std::same_as<In, f32>)
		|| (std::same_as<Out, __m128d> && std::same_as<In, f64>);

	template <typename In, typename Out>
	concept SIMDInOut256_Type = (std::same_as<Out, __m256i>
		&& (
			(std::same_as<In, u8> || std::same_as<In, i8>)
			|| (std::same_as<In, u16> || std::same_as<In, i16>)
			|| (std::same_as<In, u32> || std::same_as<In, i32>)
			|| (std::same_as<In, u64> || std::same_as<In, i64>)
			)
		)
		|| (std::same_as<Out, __m256> && std::same_as<In, f32>)
		|| (std::same_as<Out, __m256d> && std::same_as<In, f64>);

	template <typename In, typename Out>
	concept SIMDInOut_Type = ((std::same_as<Out, __m128i> || std::same_as<Out, __m256i>)
		&& (
			(std::same_as<In, u8> || std::same_as<In, i8>)
			|| (std::same_as<In, u16> || std::same_as<In, i16>)
			|| (std::same_as<In, u32> || std::same_as<In, i32>)
			|| (std::same_as<In, u64> || std::same_as<In, i64>)
			)
		)
		|| ((std::same_as<Out, __m128> || std::same_as<Out, __m256>) && std::same_as<In, f32>)
		|| ((std::same_as<Out, __m128d> || std::same_as<Out, __m256d>) && std::same_as<In, f64>);

	template <typename In>
	concept SIMDIn_Type = (std::same_as<In, u8> || std::same_as<In, i8>)
		|| (std::same_as<In, u16> || std::same_as<In, i16>)
		|| (std::same_as<In, u32> || std::same_as<In, i32>)
		|| (std::same_as<In, u64> || std::same_as<In, i64>)
		|| (std::same_as<In, f32> || std::same_as<In, f64>);

	template <typename Out>
	concept SIMDOut_Type = std::same_as<__m128, Out> || std::same_as<__m128i, Out> || std::same_as<__m128d, Out>
		|| std::same_as<__m256, Out> || std::same_as<__m256i, Out> || std::same_as<__m256d, Out>;

	/*
		It's possible to make this work with variadic number of parameters, thus constructing a 256 sized value
		from 8 32s, 16 16s, 32 8s, or 4 64s. But this cannot be done with normal variadic behavior because of default argument promotions.
		These convert smaller types like bool, char, and short into int, and float to double (which kinda messes up the size restrictions
		because sizeof will now return 4 for integrals and 8 for floatint point regardless of given type).
		Another option is variadic template arugments and ensuring they are all the same type. Seems a bit excessive, and honestly,
		a bit out of my wheel house. So just genna make 4 of these.
		Leaving this note for future me that some of these template functions could probably but a single template function
	*/
	template <typename In> requires (SIMDIn_Type<In> && sizeof(In) == 1)
		auto fill8_256(
			In a, In b, In c, In d, In e, In f, In g, In h, In i, In j, In k, In l, In m, In n, In o, In p,
			In q, In r, In s, In t, In u, In v, In w, In y, In x, In z, In aa, In ab, In ac, In ad, In ae, In af
		) -> __m256i {
		if constexpr (std::same_as<u8, In>) {
			return _mm256_set_epu8(
				a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p,
				q, r, s, t, u, v, w, y, x, z, aa, ab, ac, ad, ae, af
			);
		}
		else if constexpr (std::same_as<i8, In>) {
			return _mm256_set_epi8(
				a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p,
				q, r, s, t, u, v, w, y, x, z, aa, ab, ac, ad, ae, af
			);
		}
		else {
			std::runtime_error("Invalid Output Template Type. Only __m256i is supported.");
		}
	}

	template <typename In> requires (SIMDIn_Type<In> && sizeof(In) == 2)
		auto fill16_256(
			In a, In b, In c, In d, In e, In f, In g, In h,
			In i, In j, In k, In l, In m, In n, In o, In p
		) -> __m256i {
		if constexpr (std::same_as<u16, In>) {
			return _mm256_set_epu16(
				a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p
			);
		}
		else if constexpr (std::same_as<i16, In>) {
			return _mm256_set_epi16(
				a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p
			);
		}
		else {
			std::runtime_error("Invalid Output Template Type. Only __m256i is supported.");
		}
	}

	template <typename In> requires (SIMDIn_Type<In> && sizeof(In) == 4)
		auto fill32_256(
			In a, In b, In c, In d, In e, In f, In g, In h
		) -> std::conditional_t<std::same_as<In, f32>, __m256, __m256i> {
		if constexpr (std::same_as<f32, In>) {
			return _mm256_set_ps(a, b, c, d, e, f, g, h);
		}
		else if constexpr (std::same_as<u32, In>) {
			return _mm256_set_epu32(a, b, c, d, e, f, g, h);
		}
		else if constexpr (std::same_as<i32, In>) {
			return _mm256_set_epi32(a, b, c, d, e, f, g, h);
		}
		else {
			std::runtime_error("Invalid Types");
		}
	}

	template <typename In> requires (SIMDIn_Type<In> && sizeof(In) == 8)
		auto fill64_256(
			In a, In b, In c, In d
		) -> std::conditional_t<std::same_as<In, f64>, __m256, __m256i> {
		if constexpr (std::same_as<f64, In>) {
			return _mm256_set_pd(a, b, c, d);
		}
		else if constexpr (std::same_as<u64, In>) {
			return _mm256_set_epu64(a, b, c, d);
		}
		else if constexpr (std::same_as<i64, In>) {
			return _mm256_set_epi64(a, b, c, d);
		}
		else {
			std::runtime_error("Invalid Types");
		}
	}

	template <typename In> requires (SIMDIn_Type<In>)
		auto fill256(
			In a
		) -> std::conditional_t<std::same_as<f32, In> || std::same_as<f64, In>, __m256, __m256i> {
		if constexpr (std::same_as<u8, In>) {
			return _mm256_set_epu8(
				a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,
				a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a
			);
		}
		else if constexpr (std::same_as<i8, In>) {
			return _mm256_set_epi8(
				a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,
				a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a
			);
		}
		else if constexpr (std::same_as<u16, In>) {
			return _mm256_set_epu16(
				a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a
			);
		}
		else if constexpr (std::same_as<i16, In>) {
			return _mm256_set_epi16(
				a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a
			);
		}
		else if constexpr (std::same_as<f32, In>) {
			return _mm256_set_ps(a, a, a, a, a, a, a, a);
		}
		else if constexpr (std::same_as<u32, In>) {
			return _mm256_set_epu32(a, a, a, a, a, a, a, a);
		}
		else if constexpr (std::same_as<i32, In>) {
			return _mm256_set_epi32(a, a, a, a, a, a, a, a);
		}
		else if constexpr (std::same_as<f64, In>) {
			return _mm256_set_pd(a, a, a, a);
		}
		else if constexpr (std::same_as<u64, In>) {
			return _mm256_set_epu64(a, a, a, a);
		}
		else if constexpr (std::same_as<i64, In>) {
			return _mm256_set_epi64(a, a, a, a);
		}
		else {
			std::runtime_error("Invalid Types.");
		}
	}

	template <typename In> requires (SIMDIn_Type<In>)
		auto fill256(
			In* a
		) -> std::conditional_t<std::same_as<f32, In> || std::same_as<f64, In>, __m256, __m256i> {
		if constexpr (std::same_as<f32, In>) {
			return _mm256_load_ps((__m256*) a);
		}
		else if constexpr (std::same_as<f64, In>) {
			return _mm256_load_pd((__m256*) a);
		}
		else if constexpr (
			std::same_as<u32, In> || std::same_as<i32, In>
			|| std::same_as<u64, In> || std::same_as<i64, In>
			) {
			return _mm256_loadu_si256((__m256i*) a);
		}
		//else if constexpr (std::same_as<u32, In>) { // if AVX512F + AVX512VL
		//	return _mm256_load_epu32(a);
		//}
		//else if constexpr (std::same_as<i32, In>) { // if AVX512F + AVX512VL
		//	return _mm256_load_epi32(a);
		//}
		else {
			std::runtime_error("Invalid Types");
		}
	}

	template <typename In, typename Out> requires (SIMDInOut_Type<In, Out>)
	auto add(Out a, Out b, bool preferSaturation = false) -> Out {
		if constexpr (std::same_as<Out, __m128i>) {
			if constexpr (std::same_as<In, u8>) {
				return _mm_adds_epu8(a, b);
			}
			else if constexpr (std::same_as<In, i8>) {
				if (preferSaturation) {
					return _mm_adds_epi8(a, b);
				}
				else {
					return _mm_add_epi8(a, b);
				}
			}
			else if constexpr (std::same_as<In, u16>) {
				return _mm_adds_epu16(a, b);
			}
			else if constexpr (std::same_as<In, i16>) {
				if (preferSaturation) {
					return _mm_adds_epi16(a, b);
				}
				else {
					return _mm_add_epi16(a, b);
				}
			}
			else if constexpr (std::same_as<In, i32>) {
				return _mm_add_epi32(a, b);
			}
			else if constexpr (std::same_as<In, i64>) {
				return _mm_add_epi64(a, b);
			}
		}
		else if constexpr (std::same_as<Out, __m128> && std::same_as<In, f32>) {
			return _mm_add_ps(a, b);
		}
		else if constexpr (std::same_as<Out, __m128d> && std::same_as<In, f64>) {
			return _mm_add_pd(a, b);
		}
		else if constexpr (std::same_as<Out, __m256i>) {
			if constexpr (std::same_as<In, u8>) {
				return _mm256_adds_epu8(a, b);
			}
			else if constexpr (std::same_as<In, i8>) {
				if (preferSaturation) {
					return _mm256_adds_epi8(a, b);
				}
				else {
					return _mm256_add_epi8(a, b);
				}
			}
			else if constexpr (std::same_as<In, u16>) {
				return _mm256_adds_epu16(a, b);
			}
			else if constexpr (std::same_as<In, i16>) {
				if (preferSaturation) {
					return _mm256_adds_epi16(a, b);
				}
				else {
					return _mm256_add_epi16(a, b);
				}
			}
			else if constexpr (std::same_as<In, i32>) {
				return _mm256_add_epi32(a, b);
			}
			else if constexpr (std::same_as<In, i64>) {
				return _mm256_add_epi64(a, b);
			}
		}
		else if constexpr (std::same_as<Out, __m256> && std::same_as<In, f32>) {
			return _mm256_add_ps(a, b);
		}
		else if constexpr (std::same_as<Out, __m256d> && std::same_as<In, f64>) {
			return _mm256_add_pd(a, b);
		}
	}

	template <typename In, typename Out> requires (SIMDInOut128_Type<In, Out>)
		auto print128(Out toPrint) -> void {
		const i32 size = sizeof(Out) / sizeof(In);
		alignas(sizeof(In) * 8) In v[size];
		if constexpr (std::same_as<In, f64>) {
			_mm_storeu_pd((Out*)v, toPrint);
		}
		else if constexpr (std::same_as<In, f32>) {
			_mm_storeu_ps((Out*)v, toPrint);
		}
		else if constexpr (Integral<In>) {
			_mm_storeu_si128((Out*)v, toPrint);
		}
		std::string formatted = "{} as {}:";
		for (auto i = 0; i < size; i++) {
			formatted += " " + std::to_string(v[i]);
		}
		formatted += "\n";
		std::string typeName = typeid(In).name();
		std::cout << std::vformat(formatted, std::make_format_args(size, typeName));
	}

	template <typename In, typename Out> requires (SIMDInOut256_Type<In, Out>)
		auto print256(Out toPrint) -> void {
		const i32 size = sizeof(Out) / sizeof(In);
		alignas(sizeof(In) * 8) In v[size];
		if constexpr (std::same_as<In, f64>) {
			_mm256_storeu_pd((Out*)v, toPrint);
		}
		else if constexpr (std::same_as<In, f32>) {
			_mm256_storeu_ps((Out*)v, toPrint);
		}
		else if constexpr (Integral<In>) {
			_mm256_storeu_si256((Out*)v, toPrint);
		}
		std::string formatted = "{} as {}:";
		for (auto i = 0; i < size; i++) {
			formatted += " " + std::to_string(v[i]);
		}
		formatted += "\n";
		std::string typeName = typeid(In).name();
		std::cout << std::vformat(formatted, std::make_format_args(size, typeName));
	}

};