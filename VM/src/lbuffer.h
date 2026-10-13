// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "lobject.h"

// buffer size limit
#define MAX_BUFFER_SIZE (1 << 30)

// GCObject size has to be at least 16 bytes, so a minimum of 8 bytes is always reserved
#define sizebuffer(len) (offsetof(Buffer, data) + ((len) < 8 ? 8 : (len)))

// while C API returns 'size_t' for binary compatibility in case of future extensions,
// in the current implementation, length and offset are limited to 31 bits
// because offset is limited to an integer, a single 64bit comparison can be used and will not overflow
#define isoutofbounds(offset, len, accessize) (uint64_t(unsigned(offset)) + (accessize) > uint64_t(len))

LUAI_FUNC Buffer* luaB_newbuffer(lua_State* L, size_t s);
LUAI_FUNC void luaB_freebuffer(lua_State* L, Buffer* u, struct lua_Page* page);
