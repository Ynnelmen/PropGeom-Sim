# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

require 'kaitai/struct/struct'

unless Gem::Version.new(Kaitai::Struct::VERSION) >= Gem::Version.new('0.9')
  raise "Incompatible Kaitai Struct Ruby API: 0.9 or later is required, but you have #{Kaitai::Struct::VERSION}"
end

class OpenapiMessage < Kaitai::Struct::Struct
  def initialize(_io, _parent = nil, _root = self)
    super(_io, _parent, _root)
    _read
  end

  def _read
    @header = Header.new(@_io, self, @_root)
    case header.message_type
    when :e_message_type_e_signal_data
      @_raw_message = @_io.read_bytes(header.message_length)
      _io__raw_message = Kaitai::Struct::Stream.new(@_raw_message)
      @message = SignalData.new(_io__raw_message, self, @_root)
    when :e_message_type_e_data_quality
      @_raw_message = @_io.read_bytes(header.message_length)
      _io__raw_message = Kaitai::Struct::Stream.new(@_raw_message)
      @message = DataQuality.new(_io__raw_message, self, @_root)
    when :e_message_type_e_interpretation
      @_raw_message = @_io.read_bytes(header.message_length)
      _io__raw_message = Kaitai::Struct::Stream.new(@_raw_message)
      @message = Interpretations.new(_io__raw_message, self, @_root)
    else
      @message = @_io.read_bytes(header.message_length)
    end
    self
  end
  class Interpretations < Kaitai::Struct::Struct
    def initialize(_io, _parent = nil, _root = self)
      super(_io, _parent, _root)
      _read
    end

    def _read
      @interpretations = []
      i = 0
      while not @_io.eof?
        @interpretations << Interpretation.new(@_io, self, @_root)
        i += 1
      end
      self
    end
    attr_reader :interpretations
  end
  class DataQuality < Kaitai::Struct::Struct
    def initialize(_io, _parent = nil, _root = self)
      super(_io, _parent, _root)
      _read
    end

    def _read
      @number_of_signals = @_io.read_u2le
      @qualities = Array.new(number_of_signals)
      (number_of_signals).times { |i|
        @qualities[i] = DataQualityBlock.new(@_io, self, @_root)
      }
      self
    end
    attr_reader :number_of_signals
    attr_reader :qualities
  end
  class DataQualityBlock < Kaitai::Struct::Struct
    def initialize(_io, _parent = nil, _root = self)
      super(_io, _parent, _root)
      _read
    end

    def _read
      @signal_id = @_io.read_u2le
      @validity_flags = ValidityFlags.new(@_io, self, @_root)
      @reserved = @_io.read_u2le
      self
    end
    attr_reader :signal_id
    attr_reader :validity_flags
    attr_reader :reserved
  end
  class Interpretation < Kaitai::Struct::Struct

    E_DESCRIPTOR_TYPE = {
      1 => :e_descriptor_type_data_type,
      2 => :e_descriptor_type_scale_factor,
      3 => :e_descriptor_type_offset,
      4 => :e_descriptor_type_period_time,
      5 => :e_descriptor_type_unit,
      6 => :e_descriptor_type_vector_length,
      7 => :e_descriptor_type_channel_type,
    }
    I__E_DESCRIPTOR_TYPE = E_DESCRIPTOR_TYPE.invert
    def initialize(_io, _parent = nil, _root = self)
      super(_io, _parent, _root)
      _read
    end

    def _read
      @signal_id = @_io.read_u2le
      @descriptor_type = Kaitai::Struct::Stream::resolve_enum(E_DESCRIPTOR_TYPE, @_io.read_u2le)
      @reserved = @_io.read_u2le
      @value_length = @_io.read_u2le
      case descriptor_type
      when :e_descriptor_type_data_type
        @value = @_io.read_u2le
      when :e_descriptor_type_scale_factor
        @value = @_io.read_f8le
      when :e_descriptor_type_unit
        @value = String.new(@_io, self, @_root)
      when :e_descriptor_type_vector_length
        @value = @_io.read_u2le
      when :e_descriptor_type_period_time
        @value = Time.new(@_io, self, @_root)
      when :e_descriptor_type_offset
        @value = @_io.read_f8le
      when :e_descriptor_type_channel_type
        @value = @_io.read_u2le
      end
      @padding = Array.new(((4 - (_io.pos % 4)) & 3))
      (((4 - (_io.pos % 4)) & 3)).times { |i|
        @padding[i] = @_io.read_u1
      }
      self
    end
    attr_reader :signal_id
    attr_reader :descriptor_type
    attr_reader :reserved
    attr_reader :value_length
    attr_reader :value
    attr_reader :padding
  end
  class String < Kaitai::Struct::Struct
    def initialize(_io, _parent = nil, _root = self)
      super(_io, _parent, _root)
      _read
    end

    def _read
      @count = @_io.read_u2le
      @data = (@_io.read_bytes(count)).force_encoding("UTF-8")
      self
    end
    attr_reader :count
    attr_reader :data
  end
  class TimeFamily < Kaitai::Struct::Struct
    def initialize(_io, _parent = nil, _root = self)
      super(_io, _parent, _root)
      _read
    end

    def _read
      @k = @_io.read_u1
      @l = @_io.read_u1
      @m = @_io.read_u1
      @n = @_io.read_u1
      self
    end
    attr_reader :k
    attr_reader :l
    attr_reader :m
    attr_reader :n
  end
  class ValidityFlags < Kaitai::Struct::Struct
    def initialize(_io, _parent = nil, _root = self)
      super(_io, _parent, _root)
      _read
    end

    def _read
      @f = @_io.read_u2le
      self
    end
    def overload
      return @overload unless @overload.nil?
      @overload = (f & 2) != 0
      @overload
    end
    def invalid
      return @invalid unless @invalid.nil?
      @invalid = (f & 8) != 0
      @invalid
    end
    def overrun
      return @overrun unless @overrun.nil?
      @overrun = (f & 16) != 0
      @overrun
    end
    attr_reader :f
  end
  class SignalData < Kaitai::Struct::Struct
    def initialize(_io, _parent = nil, _root = self)
      super(_io, _parent, _root)
      _read
    end

    def _read
      @number_of_signals = @_io.read_u2le
      @reserved = @_io.read_u2le
      @signals = Array.new(number_of_signals)
      (number_of_signals).times { |i|
        @signals[i] = SignalBlock.new(@_io, self, @_root)
      }
      self
    end
    attr_reader :number_of_signals
    attr_reader :reserved
    attr_reader :signals
  end
  class Header < Kaitai::Struct::Struct

    E_MESSAGE_TYPE = {
      1 => :e_message_type_e_signal_data,
      2 => :e_message_type_e_data_quality,
      8 => :e_message_type_e_interpretation,
      11 => :e_message_type_e_aux_sequence_data,
    }
    I__E_MESSAGE_TYPE = E_MESSAGE_TYPE.invert
    def initialize(_io, _parent = nil, _root = self)
      super(_io, _parent, _root)
      _read
    end

    def _read
      @magic = @_io.read_bytes(2)
      raise Kaitai::Struct::ValidationNotEqualError.new([66, 75].pack('C*'), magic, _io, "/types/header/seq/0") if not magic == [66, 75].pack('C*')
      @header_length = @_io.read_u2le
      @message_type = Kaitai::Struct::Stream::resolve_enum(E_MESSAGE_TYPE, @_io.read_u2le)
      @reserved1 = @_io.read_u2le
      @reserved2 = @_io.read_u4le
      @time = Time.new(@_io, self, @_root)
      @message_length = @_io.read_u4le
      self
    end
    attr_reader :magic
    attr_reader :header_length
    attr_reader :message_type
    attr_reader :reserved1
    attr_reader :reserved2
    attr_reader :time
    attr_reader :message_length
  end
  class SignalBlock < Kaitai::Struct::Struct
    def initialize(_io, _parent = nil, _root = self)
      super(_io, _parent, _root)
      _read
    end

    def _read
      @signal_id = @_io.read_s2le
      @number_of_values = @_io.read_s2le
      @values = Array.new(number_of_values)
      (number_of_values).times { |i|
        @values[i] = Value.new(@_io, self, @_root)
      }
      self
    end
    attr_reader :signal_id
    attr_reader :number_of_values
    attr_reader :values
  end
  class Time < Kaitai::Struct::Struct
    def initialize(_io, _parent = nil, _root = self)
      super(_io, _parent, _root)
      _read
    end

    def _read
      @time_family = TimeFamily.new(@_io, self, @_root)
      @time_count = @_io.read_u8le
      self
    end
    attr_reader :time_family
    attr_reader :time_count
  end
  class Value < Kaitai::Struct::Struct
    def initialize(_io, _parent = nil, _root = self)
      super(_io, _parent, _root)
      _read
    end

    def _read
      @value1 = @_io.read_u1
      @value2 = @_io.read_u1
      @value3 = @_io.read_s1
      self
    end
    def calc_value
      return @calc_value unless @calc_value.nil?
      @calc_value = ((value1 + (value2 << 8)) + (value3 << 16))
      @calc_value
    end
    attr_reader :value1
    attr_reader :value2
    attr_reader :value3
  end
  attr_reader :header
  attr_reader :message
  attr_reader :_raw_message
end
