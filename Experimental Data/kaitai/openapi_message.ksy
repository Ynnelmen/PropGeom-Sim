meta:
  id: openapi_message
  endian: le

types:
  time_family:
    seq:
      - id: k
        type: u1
      - id: l
        type: u1
      - id: m
        type: u1
      - id: n
        type: u1
  time:
    seq:
      - id: time_family
        type: time_family
      - id: time_count
        type: u8
  string:
    seq:
      - id: count
        type: u2
      - id: data
        type: str
        encoding: UTF-8
        size: count

  header:
    seq:
      - id: magic
        contents: [0x42, 0x4b]
      - id: header_length
        type: u2
      - id: message_type
        type: u2
        enum: e_message_type
      - id: reserved1
        type: u2
      - id: reserved2
        type: u4
      - id: time
        type: time
      - id: message_length
        type: u4
    enums:
      e_message_type:
        1: e_signal_data
        2: e_data_quality
        8: e_interpretation
        11: e_aux_sequence_data
  
  value:
    seq:
      - id: value1
        type: u1
      - id: value2
        type: u1
      - id: value3
        type: s1
    instances:
      calc_value:
        value: value1 + (value2 << 8) + (value3 << 16)
  signal_block:
     seq:
       - id: signal_id
         type: s2
       - id: number_of_values
         type: s2
       - id: values
         type: value
         repeat: expr
         repeat-expr: number_of_values
  signal_data:
    seq:
      - id: number_of_signals
        type: u2
      - id: reserved
        type: u2
      - id: signals
        type: signal_block
        repeat: expr
        repeat-expr: number_of_signals

  validity_flags:
    seq:
      - id: f
        type: u2
    instances:
      overload:
        value: f & 2 != 0
      invalid:
        value: f & 8 != 0
      overrun:
        value: f & 16 != 0
  data_quality_block:
    seq:
      - id: signal_id
        type: u2
      - id: validity_flags
        type: validity_flags
      - id: reserved
        type: u2
  data_quality:
    seq:
      - id: number_of_signals
        type: u2
      - id: qualities
        type: data_quality_block
        repeat: expr
        repeat-expr: number_of_signals

  interpretation:
    seq:
      - id: signal_id
        type: u2
      - id: descriptor_type
        type: u2
        enum: e_descriptor_type
      - id: reserved
        type: u2
      - id: value_length
        type: u2
      - id: value
        type:
          switch-on: descriptor_type
          cases:
            'interpretation::e_descriptor_type::data_type': u2
            'interpretation::e_descriptor_type::scale_factor': f8
            'interpretation::e_descriptor_type::offset': f8
            'interpretation::e_descriptor_type::period_time': time
            'interpretation::e_descriptor_type::unit': string
            'interpretation::e_descriptor_type::vector_length': u2
            'interpretation::e_descriptor_type::channel_type': u2
      - id: padding
        type: u1
        repeat: expr
        repeat-expr: (4 - (_io.pos % 4)) & 3
    enums:
      e_descriptor_type:
        1: data_type
        2: scale_factor
        3: offset
        4: period_time
        5: unit
        6: vector_length
        7: channel_type
  interpretations:
    seq:
      - id: interpretations
        type: interpretation
        repeat: eos

seq:
  - id: header
    type: header
  - id: message
    size: header.message_length
    type: 
      switch-on: header.message_type
      cases:
        'header::e_message_type::e_signal_data': signal_data
        'header::e_message_type::e_data_quality': data_quality
        'header::e_message_type::e_interpretation': interpretations
