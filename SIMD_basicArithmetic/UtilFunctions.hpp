#pragma once

#include <type_traits>
#include <bit>

/*
	https://www.youtube.com/watch?v=SmlLdd1Q2V8
	a great video with implicit_cast and pun_cast, both very useful and great tools
*/

#define sizeofbits(x) (sizeof(x) * 8)

template <typename T>
constexpr
T implicit_cast(typename std::type_identity<T>::type val) {
	return val;
}

template <typename U, typename T>
constexpr
U pun_cast(const T& val) {
	return std::bit_cast<U>(val);
}
