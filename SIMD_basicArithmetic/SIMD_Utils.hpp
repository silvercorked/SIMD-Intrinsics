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
#include <cstring>
#include <ranges>
#include <array>

// https://learn.microsoft.com/en-us/cpp/preprocessor/predefined-macros?view=msvc-170
#ifdef _M_IX86_FP
#if _M_IX86_FP > 0
	#define __SSE__
	#if _M_IX86_FP > 1
		#define __SSE2__
	#endif
#endif
// apparently the obvious macro _M_IX86_FP contains info about the architecture of compilation targets.
// Can use this to know whether SSE and SSE2 are supposed.
// They almost always are, now-a-days, but if I'm checking for AVX, might as well do these as well.
#endif // _M_IX*^_FP // if this ain't defined, then prob genna have some issues

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

#ifdef __AVX__
	template <typename In, u32 Size> requires (SIMDIn_Type<In> && (Size == 128 || Size == 256))
	auto fillZero() {
		if constexpr (Integral<In>) {
			if constexpr (Size == 128) {
				return _mm_setzero_si128(); // __SSE2__
			}
			else if constexpr (Size == 256) {
				return _mm256_setzero_si256(); // __AVX__
			}
		}
		else if constexpr (std::same_as(f32, In)) {
			if constexpr (Size == 128) {
				return _mm_setzero_ps(); // __SSE__
			}
			else if constexpr (Size == 256) {
				return _mm256_setzero_ps();
			}
		}
		else if constexpr (std::same_as(f64, In)) {
			if constexpr (Size == 128) {
				return _mm_setzero_pd();
			}
			else if constexpr (Size == 256) {
				return _mm256_setzero_pd();
			}
		}
		else {
			std::runtime_error("Invalid Types");
		}
	}
#endif

#ifdef __AVX__
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
		In a, In b, In c, In d, In e, In f, In g, In h,
		In i, In j, In k, In l, In m, In n, In o, In p,
		In q, In r, In s, In t, In u, In v, In w, In x,
		In y, In z, In aa, In ab, In ac, In ad, In ae, In af
	) -> __m256i {
		if constexpr (std::same_as<u8, In>) {
			return _mm256_set_epi8(
				pun_cast<i8>(a), pun_cast<i8>(b), pun_cast<i8>(c), pun_cast<i8>(d),
				pun_cast<i8>(e), pun_cast<i8>(f), pun_cast<i8>(g), pun_cast<i8>(h),
				pun_cast<i8>(i), pun_cast<i8>(j), pun_cast<i8>(k), pun_cast<i8>(l),
				pun_cast<i8>(m), pun_cast<i8>(n), pun_cast<i8>(o), pun_cast<i8>(p),
				pun_cast<i8>(q), pun_cast<i8>(r), pun_cast<i8>(s), pun_cast<i8>(t),
				pun_cast<i8>(u), pun_cast<i8>(v), pun_cast<i8>(w), pun_cast<i8>(x),
				pun_cast<i8>(y), pun_cast<i8>(z), pun_cast<i8>(aa), pun_cast<i8>(ab),
				pun_cast<i8>(ac), pun_cast<i8>(ad), pun_cast<i8>(ae), pun_cast<i8>(af)
			); // least favorite function... jk _mm512_set_epi8 exists. evil
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
			return _mm256_set_epi16(
				pun_cast<i16>(a), pun_cast<i16>(b), pun_cast<i16>(c), pun_cast<i16>(d),
				pun_cast<i16>(e), pun_cast<i16>(f), pun_cast<i16>(g), pun_cast<i16>(h),
				pun_cast<i16>(i), pun_cast<i16>(j), pun_cast<i16>(k), pun_cast<i16>(l),
				pun_cast<i16>(m), pun_cast<i16>(n), pun_cast<i16>(o), pun_cast<i16>(p)
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
		In a, In b, In c, In d,
		In e, In f, In g, In h
	) -> std::conditional_t<std::same_as<In, f32>, __m256, __m256i> {
		if constexpr (std::same_as<f32, In>) {
			return _mm256_set_ps(a, b, c, d, e, f, g, h);
		}
		else if constexpr (std::same_as<u32, In>) {
			return _mm256_set_epi32(
				pun_cast<i32>(a), pun_cast<i32>(b),
				pun_cast<i32>(c), pun_cast<i32>(d),
				pun_cast<i32>(e), pun_cast<i32>(f),
				pun_cast<i32>(g), pun_cast<i32>(h)
			);
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
			return _mm256_set_epi64x(
				pun_cast<i64>(a),
				pun_cast<i64>(b),
				pun_cast<i64>(c),
				pun_cast<i64>(d)
			);
		}
		else if constexpr (std::same_as<i64, In>) {
			return _mm256_set_epi64(a, b, c, d);
		}
		else {
			std::runtime_error("Invalid Types");
		}
	}
#endif // __AVX__

#ifdef __AVX__
	template <typename In> requires (SIMDIn_Type<In>)
	auto fill256(
		In a
	) -> std::conditional_t<std::same_as<f32, In> || std::same_as<f64, In>, __m256, __m256i> {
		if constexpr (std::same_as<u8, In>) {
			return _mm256_set1_epi8(pun_cast<i8>(a)); // there are no _mm256_set1_epuXX funcs.
		}	// so type pun to int with same bits and then can still use epu ops later
		else if constexpr (std::same_as<i8, In>) {
			return _mm256_set1_epi8(a);
		}
		else if constexpr (std::same_as<u16, In>) {
			return _mm256_set1_epi16(pun_cast<i16>(a));
		}
		else if constexpr (std::same_as<i16, In>) {
			return _mm256_set1_epi16(a);
		}
		else if constexpr (std::same_as<u32, In>) {
			return _mm256_set1_epi32(pun_cast<i32>(a));
		}
		else if constexpr (std::same_as<i32, In>) {
			return _mm256_set1_epi32(a);
		}
		else if constexpr (std::same_as<u64, In>) {
			return _mm256_set1_epi64x(pun_cast<i64>(a));
		}
		else if constexpr (std::same_as<i64, In>) {
			return _mm256_set1_epi64x(a);
		}
		else if constexpr (std::same_as<f32, In>) {
			return _mm256_set1_ps(a);
		}
		else if constexpr (std::same_as<f64, In>) {
			return _mm256_set1_pd(a);
		}
		else {
			std::runtime_error("Invalid Types.");
		}
	}
#endif __AVX__

#ifdef __AVX__
	template <typename In> requires (SIMDIn_Type<In>)
	auto fill256(
		In* a
	) -> std::conditional_t<std::same_as<f32, In>, __m256, std::conditional_t<std::same_as<f64, In>, __m256d, __m256i>> {
		if constexpr (std::same_as<f32, In>) {
			return _mm256_loadu_ps(a);
		}
		else if constexpr (std::same_as<f64, In>) {
			return _mm256_loadu_pd(a);
		}
		else if constexpr (Integral<In>) {
			return _mm256_loadu_si256((__m256i*) a);
		}
		else {
			std::runtime_error("Invalid Types");
		}
	}

	/*
		Slowers version of fill256 which can be used to avoid reading outside boundaries.
		ie
		std::array<i32, SIZE> vals3;
		for (auto i = 0; i < SIZE; i++)
			vals3[i] = i;

		std::vector<__m256i> resFilling;
		for (auto i = 0; i < SIZE; i += (256 / 32)) {
			if (SIZE - i < 8) {
				resFilling.push_back(SIMD::fill256<i32>(vals3.data() + i, SIZE - i));
			}
			else {
				resFilling.push_back(SIMD::fill256<i32>(vals3.data() + i));
			}
		}
		avoids reading outside memory bound of list while also avoiding calling
		_mm256_set_epi32, which would require size dependent arguments
		ie 4 elements defined and 0 as fallback: _mm256_set_epi32(a, b, c, d, 0, 0, 0, 0)
			// no real way to generalize this for N elements defined
	*/
	template <typename In> requires (SIMDIn_Type<In>)
	auto fill256(
		In* a, size_t size, In fallback = 0
	) -> std::conditional_t<std::same_as<f32, In>, __m256, std::conditional_t<std::same_as<f64, In>, __m256d, __m256i>> {
		constexpr const auto numElems = 256 / (sizeof(In) * 8);
		std::array<In, numElems> vals;
		std::memcpy(vals.data(), a, sizeof(In) * size);
		for (auto i = size; i < numElems; i++) {
			vals[i] = fallback;
		}
		return fill256<In>(vals.data());
	}

	/*
		Fills based on a mask. Only uses N sequential elements of input list where N = bitcount(mask).
		Only first M bits of mask are used where M is result type size (256) / sizeof(In) * 8.

		useful for multiplication cases:
		ex
		_mm256_mul_epi32 multiplies only the lower 128 bits of input a and b to produce the output 256 bits
		a: 0 0 0 0 s t u v <-- one non-space token is 32 bits
		b: 0 0 0 0 w x y z
		out: s*w t*x u*y v*z <-- one non-space token in 64 bits
		can create a and b with this function like
		auto a = fillMask256(src, 0x0F);

		can also make more complex formulations like:
		0 s 0 t 0 u 0 v
	*/
	template <typename In> requires (SIMDIn_Type<In>)
	auto fillMask256(
		In* a, u32 mask, In fallback = 0
	) -> std::conditional_t<std::same_as<f32, In>, __m256, std::conditional_t<std::same_as<f64, In>, __m256d, __m256i>> {
		constexpr const auto numElems = 256 / (sizeof(In) * 8);
		std::array<In, numElems> vals;
		size_t offset = 0;
		for (auto i = 0; i < numElems; i++) {
			if (mask & (1 << i))
				vals[i] = a[offset++];
			else
				vals[i] = fallback;
		}
		return fill256<In>(vals.data());
	}
#endif // __AVX__

#ifdef __AVX2__
	template <typename In, typename Out> requires (SIMDInOut_Type<In, Out>)
	auto add(const Out& a, const Out& b, bool preferSaturation = false) -> Out {
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

	template <typename In, typename Out> requires (SIMDInOut_Type<In, Out>)
	auto subtract(const Out& a, const Out& b, bool preferSaturation = false) -> Out {
		if constexpr (std::same_as<Out, __m128i>) {
			if constexpr (std::same_as<In, u8>) {
				return _mm_subs_epu8(a, b);
			}
			else if constexpr (std::same_as<In, i8>) {
				if (preferSaturation) {
					return _mm_subs_epi8(a, b);
				}
				else {
					return _mm_sub_epi8(a, b);
				}
			}
			else if constexpr (std::same_as<In, u16>) {
				return _mm_subs_epu16(a, b);
			}
			else if constexpr (std::same_as<In, i16>) {
				if (preferSaturation) {
					return _mm_subs_epi16(a, b);
				}
				else {
					return _mm_sub_epi16(a, b);
				}
			}
			else if constexpr (std::same_as<In, i32>) {
				return _mm_sub_epi32(a, b);
			}
			else if constexpr (std::same_as<In, i64>) {
				return _mm_sub_epi64(a, b);
			}
		}
		else if constexpr (std::same_as<Out, __m128> && std::same_as<In, f32>) {
			return _mm_sub_ps(a, b);
		}
		else if constexpr (std::same_as<Out, __m128d> && std::same_as<In, f64>) {
			return _mm_sub_pd(a, b);
		}
		else if constexpr (std::same_as<Out, __m256i>) {
			if constexpr (std::same_as<In, u8>) {
				return _mm256_subs_epu8(a, b);
			}
			else if constexpr (std::same_as<In, i8>) {
				if (preferSaturation) {
					return _mm256_subs_epi8(a, b);
				}
				else {
					return _mm256_sub_epi8(a, b);
				}
			}
			else if constexpr (std::same_as<In, u16>) {
				return _mm256_subs_epu16(a, b);
			}
			else if constexpr (std::same_as<In, i16>) {
				if (preferSaturation) {
					return _mm256_subs_epi16(a, b);
				}
				else {
					return _mm256_sub_epi16(a, b);
				}
			}
			else if constexpr (std::same_as<In, i32>) {
				return _mm256_sub_epi32(a, b);
			}
			else if constexpr (std::same_as<In, i64>) {
				return _mm256_sub_epi64(a, b);
			}
		}
		else if constexpr (std::same_as<Out, __m256> && std::same_as<In, f32>) {
			return _mm256_sub_ps(a, b);
		}
		else if constexpr (std::same_as<Out, __m256d> && std::same_as<In, f64>) {
			return _mm256_sub_pd(a, b);
		}
	}
#endif // __AVX2__

#ifdef __AVX512F__
	/*
		for 128 bit types, normal add has better CPI (throughput) than masked versions. But for 256, they're the same.
		Without setting mask or src during call, function behaves the same as add, but can be used to only do arithmetic
		on sub set of elements within (ie, 16 can fit in 128, but if only 10 filled, can just do 10 with mask 0x0000 03FF
		(ie 0b 0000 0000 0000 0000  0000 0011 1111 1111).
	*/
	template <typename In, typename Out> requires (SIMDInOut_Type<In, Out>)
	auto maskAdd(
		const Out& a,
		const Out& b,
		u32 mask = std::numeric_limits<u32>::max(),
		const Out& src = fillZero<In, sizeof(Out) * 8>(), // temporary allowed cause const ref. if just ref, not allowed
		bool preferSaturation = false
	) -> Out {
		if constexpr (std::same_as<Out, __m128i>) {
			if constexpr (std::same_as<In, u8>) {
				return _mm_mask_adds_epu8(src, static_cast<u16>(mask), a, b);
			}
			else if constexpr (std::same_as<In, i8>) {
				if (preferSaturation) {
					return _mm_mask_adds_epi8(src, static_cast<u16>(mask), a, b);
				}
				else {
					return _mm_mask_add_epi8(src, static_cast<u16>(mask), a, b);
				}
			}
			else if constexpr (std::same_as<In, u16>) {
				return _mm_mask_adds_epu16(src, static_cast<u8>(mask), a, b);
			}
			else if constexpr (std::same_as<In, i16>) {
				if (preferSaturation) {
					return _mm_mask_adds_epi16(src, static_cast<u8>(mask), a, b);
				}
				else {
					return _mm_mask_add_epi16(src, static_cast<u8>(mask), a, b);
				}
			}
			else if constexpr (std::same_as<In, i32>) {
				return _mm_mask_add_epi32(src, static_cast<u8>(mask), a, b);
			}
			else if constexpr (std::same_as<In, i64>) {
				return _mm_mask_add_epi64(src, static_cast<u8>(mask), a, b);
			}
		}
		else if constexpr (std::same_as<Out, __m128> && std::same_as<In, f32>) {
			return _mm_mask_add_ps(src, static_cast<u8>(mask), a, b);
		}
		else if constexpr (std::same_as<Out, __m128d> && std::same_as<In, f64>) {
			return _mm_mask_add_pd(src, static_cast<u8>(mask), a, b);
		}
		else if constexpr (std::same_as<Out, __m256i>) {
			if constexpr (std::same_as<In, u8>) {
				return _mm256_mask_adds_epu8(src, mask, a, b);
			}
			else if constexpr (std::same_as<In, i8>) {
				if (preferSaturation) {
					return _mm256_mask_adds_epi8(src, mask, a, b);
				}
				else {
					return _mm256_mask_add_epi8(src, mask, a, b);
				}
			}
			else if constexpr (std::same_as<In, u16>) {
				return _mm256_mask_adds_epu16(src, static_cast<u16>(mask), a, b);
			}
			else if constexpr (std::same_as<In, i16>) {
				if (preferSaturation) {
					return _mm256_mask_adds_epi16(src, static_cast<u16>(mask), a, b);
				}
				else {
					return _mm256_mask_add_epi16(src, static_cast<u16>(mask), a, b);
				}
			}
			else if constexpr (std::same_as<In, i32>) {
				return _mm256_mask_add_epi32(src, static_cast<u8>(mask), a, b);
			}
			else if constexpr (std::same_as<In, i64>) {
				return _mm256_mask_add_epi64(src, static_cast<u8>(mask), a, b);
			}
		}
		else if constexpr (std::same_as<Out, __m256> && std::same_as<In, f32>) {
			return _mm256_mask_add_ps(src, static_cast<u8>(mask), a, b);
		}
		else if constexpr (std::same_as<Out, __m256d> && std::same_as<In, f64>) {
			return _mm256_mask_add_pd(src, static_cast<u8>(mask), a, b);
		}
	}

	template <typename In, typename Out> requires (SIMDInOut_Type<In, Out>)
	auto maskSubtract(
		const Out& a,
		const Out& b,
		u32 mask = std::numeric_limits<u32>::max(),
		const Out& src = fillZero<In, sizeof(Out) * 8>(),
		bool preferSaturation = false
	) -> Out {
		if constexpr (std::same_as<Out, __m128i>) {
			if constexpr (std::same_as<In, u8>) {
				return _mm_mask_subs_epu8(src, static_cast<u16>(mask), a, b);
			}
			else if constexpr (std::same_as<In, i8>) {
				if (preferSaturation) {
					return _mm_mask_subs_epi8(src, static_cast<u16>(mask), a, b);
				}
				else {
					return _mm_mask_sub_epi8(src, static_cast<u16>(mask), a, b);
				}
			}
			else if constexpr (std::same_as<In, u16>) {
				return _mm_mask_subs_epu16(src, static_cast<u8>(mask), a, b);
			}
			else if constexpr (std::same_as<In, i16>) {
				if (preferSaturation) {
					return _mm_mask_subs_epi16(src, static_cast<u8>(mask), a, b);
				}
				else {
					return _mm_mask_sub_epi16(src, static_cast<u8>(mask), a, b);
				}
			}
			else if constexpr (std::same_as<In, i32>) {
				return _mm_mask_sub_epi32(src, static_cast<u8>(mask), a, b);
			}
			else if constexpr (std::same_as<In, i64>) {
				return _mm_mask_sub_epi64(src, static_cast<u8>(mask), a, b);
			}
		}
		else if constexpr (std::same_as<Out, __m128> && std::same_as<In, f32>) {
			return _mm_mask_sub_ps(src, static_cast<u8>(mask), a, b);
		}
		else if constexpr (std::same_as<Out, __m128d> && std::same_as<In, f64>) {
			return _mm_mask_sub_pd(src, static_cast<u8>(mask), a, b);
		}
		else if constexpr (std::same_as<Out, __m256i>) {
			if constexpr (std::same_as<In, u8>) {
				return _mm256_mask_subs_epu8(src, mask, a, b);
			}
			else if constexpr (std::same_as<In, i8>) {
				if (preferSaturation) {
					return _mm256_mask_subs_epi8(src, mask, a, b);
				}
				else {
					return _mm256_mask_sub_epi8(src, mask, a, b);
				}
			}
			else if constexpr (std::same_as<In, u16>) {
				return _mm256_mask_subs_epu16(src, static_cast<u16>(mask), a, b);
			}
			else if constexpr (std::same_as<In, i16>) {
				if (preferSaturation) {
					return _mm256_mask_subs_epi16(src, static_cast<u16>(mask), a, b);
				}
				else {
					return _mm256_mask_sub_epi16(src, static_cast<u16>(mask), a, b);
				}
			}
			else if constexpr (std::same_as<In, i32>) {
				return _mm256_mask_sub_epi32(src, static_cast<u8>(mask), a, b);
			}
			else if constexpr (std::same_as<In, i64>) {
				return _mm256_mask_sub_epi64(src, static_cast<u8>(mask), a, b);
			}
		}
		else if constexpr (std::same_as<Out, __m256> && std::same_as<In, f32>) {
			return _mm256_mask_sub_ps(src, static_cast<u8>(mask), a, b);
		}
		else if constexpr (std::same_as<Out, __m256d> && std::same_as<In, f64>) {
			return _mm256_mask_sub_pd(src, static_cast<u8>(mask), a, b);
		}
	}
#endif // __AVX512F__

#ifdef __SSE2__
	template <typename In, typename Out> requires (SIMDInOut128_Type<In, Out>)
	auto print128(const Out& toPrint) -> void {
		const i32 size = sizeof(Out) / sizeof(In);
		alignas(sizeof(In) * 8) In v[size];
		if constexpr (std::same_as<In, f64>) {
			_mm_storeu_pd((Out*)v, toPrint);
		}
		else if constexpr (std::same_as<In, f32>) {
			_mm_storeu_ps((Out*)v, toPrint); // just __SSE__
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
#endif // __SSE2__

#ifdef __AVX__
	template <typename In, typename Out> requires (SIMDInOut256_Type<In, Out>)
		auto print256(const Out& toPrint) -> void {
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
#endif // __AVX__
};