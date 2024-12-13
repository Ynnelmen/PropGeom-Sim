-- This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild
--
-- This file is compatible with Lua 5.3

local class = require("class")
require("kaitaistruct")
local stringstream = require("string_stream")
local enum = require("enum")
local str_decode = require("string_decode")

OpenapiMessage = class.class(KaitaiStruct)

function OpenapiMessage:_init(io, parent, root)
  KaitaiStruct._init(self, io)
  self._parent = parent
  self._root = root or self
  self:_read()
end

function OpenapiMessage:_read()
  self.header = OpenapiMessage.Header(self._io, self, self._root)
  local _on = self.header.message_type
  if _on == OpenapiMessage.Header.EMessageType.e_signal_data then
    self._raw_message = self._io:read_bytes(self.header.message_length)
    local _io = KaitaiStream(stringstream(self._raw_message))
    self.message = OpenapiMessage.SignalData(_io, self, self._root)
  elseif _on == OpenapiMessage.Header.EMessageType.e_data_quality then
    self._raw_message = self._io:read_bytes(self.header.message_length)
    local _io = KaitaiStream(stringstream(self._raw_message))
    self.message = OpenapiMessage.DataQuality(_io, self, self._root)
  elseif _on == OpenapiMessage.Header.EMessageType.e_interpretation then
    self._raw_message = self._io:read_bytes(self.header.message_length)
    local _io = KaitaiStream(stringstream(self._raw_message))
    self.message = OpenapiMessage.Interpretations(_io, self, self._root)
  else
    self.message = self._io:read_bytes(self.header.message_length)
  end
end


OpenapiMessage.Interpretations = class.class(KaitaiStruct)

function OpenapiMessage.Interpretations:_init(io, parent, root)
  KaitaiStruct._init(self, io)
  self._parent = parent
  self._root = root or self
  self:_read()
end

function OpenapiMessage.Interpretations:_read()
  self.interpretations = {}
  local i = 0
  while not self._io:is_eof() do
    self.interpretations[i + 1] = OpenapiMessage.Interpretation(self._io, self, self._root)
    i = i + 1
  end
end


OpenapiMessage.DataQuality = class.class(KaitaiStruct)

function OpenapiMessage.DataQuality:_init(io, parent, root)
  KaitaiStruct._init(self, io)
  self._parent = parent
  self._root = root or self
  self:_read()
end

function OpenapiMessage.DataQuality:_read()
  self.number_of_signals = self._io:read_u2le()
  self.qualities = {}
  for i = 0, self.number_of_signals - 1 do
    self.qualities[i + 1] = OpenapiMessage.DataQualityBlock(self._io, self, self._root)
  end
end


OpenapiMessage.DataQualityBlock = class.class(KaitaiStruct)

function OpenapiMessage.DataQualityBlock:_init(io, parent, root)
  KaitaiStruct._init(self, io)
  self._parent = parent
  self._root = root or self
  self:_read()
end

function OpenapiMessage.DataQualityBlock:_read()
  self.signal_id = self._io:read_u2le()
  self.validity_flags = OpenapiMessage.ValidityFlags(self._io, self, self._root)
  self.reserved = self._io:read_u2le()
end


OpenapiMessage.Interpretation = class.class(KaitaiStruct)

OpenapiMessage.Interpretation.EDescriptorType = enum.Enum {
  data_type = 1,
  scale_factor = 2,
  offset = 3,
  period_time = 4,
  unit = 5,
  vector_length = 6,
  channel_type = 7,
}

function OpenapiMessage.Interpretation:_init(io, parent, root)
  KaitaiStruct._init(self, io)
  self._parent = parent
  self._root = root or self
  self:_read()
end

function OpenapiMessage.Interpretation:_read()
  self.signal_id = self._io:read_u2le()
  self.descriptor_type = OpenapiMessage.Interpretation.EDescriptorType(self._io:read_u2le())
  self.reserved = self._io:read_u2le()
  self.value_length = self._io:read_u2le()
  local _on = self.descriptor_type
  if _on == OpenapiMessage.Interpretation.EDescriptorType.data_type then
    self.value = self._io:read_u2le()
  elseif _on == OpenapiMessage.Interpretation.EDescriptorType.scale_factor then
    self.value = self._io:read_f8le()
  elseif _on == OpenapiMessage.Interpretation.EDescriptorType.unit then
    self.value = OpenapiMessage.String(self._io, self, self._root)
  elseif _on == OpenapiMessage.Interpretation.EDescriptorType.vector_length then
    self.value = self._io:read_u2le()
  elseif _on == OpenapiMessage.Interpretation.EDescriptorType.period_time then
    self.value = OpenapiMessage.Time(self._io, self, self._root)
  elseif _on == OpenapiMessage.Interpretation.EDescriptorType.offset then
    self.value = self._io:read_f8le()
  elseif _on == OpenapiMessage.Interpretation.EDescriptorType.channel_type then
    self.value = self._io:read_u2le()
  end
  self.padding = {}
  for i = 0, ((4 - (self._io:pos() % 4)) & 3) - 1 do
    self.padding[i + 1] = self._io:read_u1()
  end
end


OpenapiMessage.String = class.class(KaitaiStruct)

function OpenapiMessage.String:_init(io, parent, root)
  KaitaiStruct._init(self, io)
  self._parent = parent
  self._root = root or self
  self:_read()
end

function OpenapiMessage.String:_read()
  self.count = self._io:read_u2le()
  self.data = str_decode.decode(self._io:read_bytes(self.count), "UTF-8")
end


OpenapiMessage.TimeFamily = class.class(KaitaiStruct)

function OpenapiMessage.TimeFamily:_init(io, parent, root)
  KaitaiStruct._init(self, io)
  self._parent = parent
  self._root = root or self
  self:_read()
end

function OpenapiMessage.TimeFamily:_read()
  self.k = self._io:read_u1()
  self.l = self._io:read_u1()
  self.m = self._io:read_u1()
  self.n = self._io:read_u1()
end


OpenapiMessage.ValidityFlags = class.class(KaitaiStruct)

function OpenapiMessage.ValidityFlags:_init(io, parent, root)
  KaitaiStruct._init(self, io)
  self._parent = parent
  self._root = root or self
  self:_read()
end

function OpenapiMessage.ValidityFlags:_read()
  self.f = self._io:read_u2le()
end

OpenapiMessage.ValidityFlags.property.overload = {}
function OpenapiMessage.ValidityFlags.property.overload:get()
  if self._m_overload ~= nil then
    return self._m_overload
  end

  self._m_overload = (self.f & 2) ~= 0
  return self._m_overload
end

OpenapiMessage.ValidityFlags.property.invalid = {}
function OpenapiMessage.ValidityFlags.property.invalid:get()
  if self._m_invalid ~= nil then
    return self._m_invalid
  end

  self._m_invalid = (self.f & 8) ~= 0
  return self._m_invalid
end

OpenapiMessage.ValidityFlags.property.overrun = {}
function OpenapiMessage.ValidityFlags.property.overrun:get()
  if self._m_overrun ~= nil then
    return self._m_overrun
  end

  self._m_overrun = (self.f & 16) ~= 0
  return self._m_overrun
end


OpenapiMessage.SignalData = class.class(KaitaiStruct)

function OpenapiMessage.SignalData:_init(io, parent, root)
  KaitaiStruct._init(self, io)
  self._parent = parent
  self._root = root or self
  self:_read()
end

function OpenapiMessage.SignalData:_read()
  self.number_of_signals = self._io:read_u2le()
  self.reserved = self._io:read_u2le()
  self.signals = {}
  for i = 0, self.number_of_signals - 1 do
    self.signals[i + 1] = OpenapiMessage.SignalBlock(self._io, self, self._root)
  end
end


OpenapiMessage.Header = class.class(KaitaiStruct)

OpenapiMessage.Header.EMessageType = enum.Enum {
  e_signal_data = 1,
  e_data_quality = 2,
  e_interpretation = 8,
  e_aux_sequence_data = 11,
}

function OpenapiMessage.Header:_init(io, parent, root)
  KaitaiStruct._init(self, io)
  self._parent = parent
  self._root = root or self
  self:_read()
end

function OpenapiMessage.Header:_read()
  self.magic = self._io:read_bytes(2)
  if not(self.magic == "\066\075") then
    error("not equal, expected " ..  "\066\075" .. ", but got " .. self.magic)
  end
  self.header_length = self._io:read_u2le()
  self.message_type = OpenapiMessage.Header.EMessageType(self._io:read_u2le())
  self.reserved1 = self._io:read_u2le()
  self.reserved2 = self._io:read_u4le()
  self.time = OpenapiMessage.Time(self._io, self, self._root)
  self.message_length = self._io:read_u4le()
end


OpenapiMessage.SignalBlock = class.class(KaitaiStruct)

function OpenapiMessage.SignalBlock:_init(io, parent, root)
  KaitaiStruct._init(self, io)
  self._parent = parent
  self._root = root or self
  self:_read()
end

function OpenapiMessage.SignalBlock:_read()
  self.signal_id = self._io:read_s2le()
  self.number_of_values = self._io:read_s2le()
  self.values = {}
  for i = 0, self.number_of_values - 1 do
    self.values[i + 1] = OpenapiMessage.Value(self._io, self, self._root)
  end
end


OpenapiMessage.Time = class.class(KaitaiStruct)

function OpenapiMessage.Time:_init(io, parent, root)
  KaitaiStruct._init(self, io)
  self._parent = parent
  self._root = root or self
  self:_read()
end

function OpenapiMessage.Time:_read()
  self.time_family = OpenapiMessage.TimeFamily(self._io, self, self._root)
  self.time_count = self._io:read_u8le()
end


OpenapiMessage.Value = class.class(KaitaiStruct)

function OpenapiMessage.Value:_init(io, parent, root)
  KaitaiStruct._init(self, io)
  self._parent = parent
  self._root = root or self
  self:_read()
end

function OpenapiMessage.Value:_read()
  self.value1 = self._io:read_u1()
  self.value2 = self._io:read_u1()
  self.value3 = self._io:read_s1()
end

OpenapiMessage.Value.property.calc_value = {}
function OpenapiMessage.Value.property.calc_value:get()
  if self._m_calc_value ~= nil then
    return self._m_calc_value
  end

  self._m_calc_value = ((self.value1 + (self.value2 << 8)) + (self.value3 << 16))
  return self._m_calc_value
end


