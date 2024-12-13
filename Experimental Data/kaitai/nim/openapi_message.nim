import kaitai_struct_nim_runtime
import options

type
  OpenapiMessage* = ref object of KaitaiStruct
    `header`*: OpenapiMessage_Header
    `message`*: KaitaiStruct
    `parent`*: KaitaiStruct
    `rawMessage`*: seq[byte]
  OpenapiMessage_Interpretations* = ref object of KaitaiStruct
    `interpretations`*: seq[OpenapiMessage_Interpretation]
    `parent`*: OpenapiMessage
  OpenapiMessage_DataQuality* = ref object of KaitaiStruct
    `numberOfSignals`*: uint16
    `qualities`*: seq[OpenapiMessage_DataQualityBlock]
    `parent`*: OpenapiMessage
  OpenapiMessage_DataQualityBlock* = ref object of KaitaiStruct
    `signalId`*: uint16
    `validityFlags`*: OpenapiMessage_ValidityFlags
    `reserved`*: uint16
    `parent`*: OpenapiMessage_DataQuality
  OpenapiMessage_Interpretation* = ref object of KaitaiStruct
    `signalId`*: uint16
    `descriptorType`*: OpenapiMessage_Interpretation_EDescriptorType
    `reserved`*: uint16
    `valueLength`*: uint16
    `value`*: KaitaiStruct
    `padding`*: seq[uint8]
    `parent`*: OpenapiMessage_Interpretations
  OpenapiMessage_Interpretation_EDescriptorType* = enum
    data_type = 1
    scale_factor = 2
    offset = 3
    period_time = 4
    unit = 5
    vector_length = 6
    channel_type = 7
  OpenapiMessage_String* = ref object of KaitaiStruct
    `count`*: uint16
    `data`*: string
    `parent`*: OpenapiMessage_Interpretation
  OpenapiMessage_TimeFamily* = ref object of KaitaiStruct
    `k`*: uint8
    `l`*: uint8
    `m`*: uint8
    `n`*: uint8
    `parent`*: OpenapiMessage_Time
  OpenapiMessage_ValidityFlags* = ref object of KaitaiStruct
    `f`*: uint16
    `parent`*: OpenapiMessage_DataQualityBlock
    `overloadInst`*: bool
    `invalidInst`*: bool
    `overrunInst`*: bool
  OpenapiMessage_SignalData* = ref object of KaitaiStruct
    `numberOfSignals`*: uint16
    `reserved`*: uint16
    `signals`*: seq[OpenapiMessage_SignalBlock]
    `parent`*: OpenapiMessage
  OpenapiMessage_Header* = ref object of KaitaiStruct
    `magic`*: seq[byte]
    `headerLength`*: uint16
    `messageType`*: OpenapiMessage_Header_EMessageType
    `reserved1`*: uint16
    `reserved2`*: uint32
    `time`*: OpenapiMessage_Time
    `messageLength`*: uint32
    `parent`*: OpenapiMessage
  OpenapiMessage_Header_EMessageType* = enum
    e_signal_data = 1
    e_data_quality = 2
    e_interpretation = 8
    e_aux_sequence_data = 11
  OpenapiMessage_SignalBlock* = ref object of KaitaiStruct
    `signalId`*: int16
    `numberOfValues`*: int16
    `values`*: seq[OpenapiMessage_Value]
    `parent`*: OpenapiMessage_SignalData
  OpenapiMessage_Time* = ref object of KaitaiStruct
    `timeFamily`*: OpenapiMessage_TimeFamily
    `timeCount`*: uint64
    `parent`*: KaitaiStruct
  OpenapiMessage_Value* = ref object of KaitaiStruct
    `value1`*: uint8
    `value2`*: uint8
    `value3`*: int8
    `parent`*: OpenapiMessage_SignalBlock
    `calcValueInst`*: int

proc read*(_: typedesc[OpenapiMessage], io: KaitaiStream, root: KaitaiStruct, parent: KaitaiStruct): OpenapiMessage
proc read*(_: typedesc[OpenapiMessage_Interpretations], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage): OpenapiMessage_Interpretations
proc read*(_: typedesc[OpenapiMessage_DataQuality], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage): OpenapiMessage_DataQuality
proc read*(_: typedesc[OpenapiMessage_DataQualityBlock], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage_DataQuality): OpenapiMessage_DataQualityBlock
proc read*(_: typedesc[OpenapiMessage_Interpretation], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage_Interpretations): OpenapiMessage_Interpretation
proc read*(_: typedesc[OpenapiMessage_String], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage_Interpretation): OpenapiMessage_String
proc read*(_: typedesc[OpenapiMessage_TimeFamily], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage_Time): OpenapiMessage_TimeFamily
proc read*(_: typedesc[OpenapiMessage_ValidityFlags], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage_DataQualityBlock): OpenapiMessage_ValidityFlags
proc read*(_: typedesc[OpenapiMessage_SignalData], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage): OpenapiMessage_SignalData
proc read*(_: typedesc[OpenapiMessage_Header], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage): OpenapiMessage_Header
proc read*(_: typedesc[OpenapiMessage_SignalBlock], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage_SignalData): OpenapiMessage_SignalBlock
proc read*(_: typedesc[OpenapiMessage_Time], io: KaitaiStream, root: KaitaiStruct, parent: KaitaiStruct): OpenapiMessage_Time
proc read*(_: typedesc[OpenapiMessage_Value], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage_SignalBlock): OpenapiMessage_Value

proc overload*(this: OpenapiMessage_ValidityFlags): bool
proc invalid*(this: OpenapiMessage_ValidityFlags): bool
proc overrun*(this: OpenapiMessage_ValidityFlags): bool
proc calcValue*(this: OpenapiMessage_Value): int

proc read*(_: typedesc[OpenapiMessage], io: KaitaiStream, root: KaitaiStruct, parent: KaitaiStruct): OpenapiMessage =
  template this: untyped = result
  this = new(OpenapiMessage)
  let root = if root == nil: cast[OpenapiMessage](this) else: cast[OpenapiMessage](root)
  this.io = io
  this.root = root
  this.parent = parent

  let headerExpr = OpenapiMessage_Header.read(this.io, this.root, this)
  this.header = headerExpr
  block:
    let on = this.header.messageType
    if on == openapi_message.e_signal_data:
      let rawMessageExpr = this.io.readBytes(int(this.header.messageLength))
      this.rawMessage = rawMessageExpr
      let rawMessageIo = newKaitaiStream(rawMessageExpr)
      let messageExpr = OpenapiMessage_SignalData.read(rawMessageIo, this.root, this)
      this.message = messageExpr
    elif on == openapi_message.e_data_quality:
      let rawMessageExpr = this.io.readBytes(int(this.header.messageLength))
      this.rawMessage = rawMessageExpr
      let rawMessageIo = newKaitaiStream(rawMessageExpr)
      let messageExpr = OpenapiMessage_DataQuality.read(rawMessageIo, this.root, this)
      this.message = messageExpr
    elif on == openapi_message.e_interpretation:
      let rawMessageExpr = this.io.readBytes(int(this.header.messageLength))
      this.rawMessage = rawMessageExpr
      let rawMessageIo = newKaitaiStream(rawMessageExpr)
      let messageExpr = OpenapiMessage_Interpretations.read(rawMessageIo, this.root, this)
      this.message = messageExpr
    else:
      let messageExpr = this.io.readBytes(int(this.header.messageLength))
      this.message = messageExpr

proc fromFile*(_: typedesc[OpenapiMessage], filename: string): OpenapiMessage =
  OpenapiMessage.read(newKaitaiFileStream(filename), nil, nil)

proc read*(_: typedesc[OpenapiMessage_Interpretations], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage): OpenapiMessage_Interpretations =
  template this: untyped = result
  this = new(OpenapiMessage_Interpretations)
  let root = if root == nil: cast[OpenapiMessage](this) else: cast[OpenapiMessage](root)
  this.io = io
  this.root = root
  this.parent = parent

  block:
    var i: int
    while not this.io.isEof:
      let it = OpenapiMessage_Interpretation.read(this.io, this.root, this)
      this.interpretations.add(it)
      inc i

proc fromFile*(_: typedesc[OpenapiMessage_Interpretations], filename: string): OpenapiMessage_Interpretations =
  OpenapiMessage_Interpretations.read(newKaitaiFileStream(filename), nil, nil)

proc read*(_: typedesc[OpenapiMessage_DataQuality], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage): OpenapiMessage_DataQuality =
  template this: untyped = result
  this = new(OpenapiMessage_DataQuality)
  let root = if root == nil: cast[OpenapiMessage](this) else: cast[OpenapiMessage](root)
  this.io = io
  this.root = root
  this.parent = parent

  let numberOfSignalsExpr = this.io.readU2le()
  this.numberOfSignals = numberOfSignalsExpr
  for i in 0 ..< int(this.numberOfSignals):
    let it = OpenapiMessage_DataQualityBlock.read(this.io, this.root, this)
    this.qualities.add(it)

proc fromFile*(_: typedesc[OpenapiMessage_DataQuality], filename: string): OpenapiMessage_DataQuality =
  OpenapiMessage_DataQuality.read(newKaitaiFileStream(filename), nil, nil)

proc read*(_: typedesc[OpenapiMessage_DataQualityBlock], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage_DataQuality): OpenapiMessage_DataQualityBlock =
  template this: untyped = result
  this = new(OpenapiMessage_DataQualityBlock)
  let root = if root == nil: cast[OpenapiMessage](this) else: cast[OpenapiMessage](root)
  this.io = io
  this.root = root
  this.parent = parent

  let signalIdExpr = this.io.readU2le()
  this.signalId = signalIdExpr
  let validityFlagsExpr = OpenapiMessage_ValidityFlags.read(this.io, this.root, this)
  this.validityFlags = validityFlagsExpr
  let reservedExpr = this.io.readU2le()
  this.reserved = reservedExpr

proc fromFile*(_: typedesc[OpenapiMessage_DataQualityBlock], filename: string): OpenapiMessage_DataQualityBlock =
  OpenapiMessage_DataQualityBlock.read(newKaitaiFileStream(filename), nil, nil)

proc read*(_: typedesc[OpenapiMessage_Interpretation], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage_Interpretations): OpenapiMessage_Interpretation =
  template this: untyped = result
  this = new(OpenapiMessage_Interpretation)
  let root = if root == nil: cast[OpenapiMessage](this) else: cast[OpenapiMessage](root)
  this.io = io
  this.root = root
  this.parent = parent

  let signalIdExpr = this.io.readU2le()
  this.signalId = signalIdExpr
  let descriptorTypeExpr = OpenapiMessage_Interpretation_EDescriptorType(this.io.readU2le())
  this.descriptorType = descriptorTypeExpr
  let reservedExpr = this.io.readU2le()
  this.reserved = reservedExpr
  let valueLengthExpr = this.io.readU2le()
  this.valueLength = valueLengthExpr
  block:
    let on = this.descriptorType
    if on == openapi_message.data_type:
      let valueExpr = KaitaiStruct(this.io.readU2le())
      this.value = valueExpr
    elif on == openapi_message.scale_factor:
      let valueExpr = KaitaiStruct(this.io.readF8le())
      this.value = valueExpr
    elif on == openapi_message.unit:
      let valueExpr = OpenapiMessage_String.read(this.io, this.root, this)
      this.value = valueExpr
    elif on == openapi_message.vector_length:
      let valueExpr = KaitaiStruct(this.io.readU2le())
      this.value = valueExpr
    elif on == openapi_message.period_time:
      let valueExpr = OpenapiMessage_Time.read(this.io, this.root, this)
      this.value = valueExpr
    elif on == openapi_message.offset:
      let valueExpr = KaitaiStruct(this.io.readF8le())
      this.value = valueExpr
    elif on == openapi_message.channel_type:
      let valueExpr = KaitaiStruct(this.io.readU2le())
      this.value = valueExpr
  for i in 0 ..< int(((4 - (this.io.pos %%% 4)) and 3)):
    let it = this.io.readU1()
    this.padding.add(it)

proc fromFile*(_: typedesc[OpenapiMessage_Interpretation], filename: string): OpenapiMessage_Interpretation =
  OpenapiMessage_Interpretation.read(newKaitaiFileStream(filename), nil, nil)

proc read*(_: typedesc[OpenapiMessage_String], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage_Interpretation): OpenapiMessage_String =
  template this: untyped = result
  this = new(OpenapiMessage_String)
  let root = if root == nil: cast[OpenapiMessage](this) else: cast[OpenapiMessage](root)
  this.io = io
  this.root = root
  this.parent = parent

  let countExpr = this.io.readU2le()
  this.count = countExpr
  let dataExpr = encode(this.io.readBytes(int(this.count)), "UTF-8")
  this.data = dataExpr

proc fromFile*(_: typedesc[OpenapiMessage_String], filename: string): OpenapiMessage_String =
  OpenapiMessage_String.read(newKaitaiFileStream(filename), nil, nil)

proc read*(_: typedesc[OpenapiMessage_TimeFamily], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage_Time): OpenapiMessage_TimeFamily =
  template this: untyped = result
  this = new(OpenapiMessage_TimeFamily)
  let root = if root == nil: cast[OpenapiMessage](this) else: cast[OpenapiMessage](root)
  this.io = io
  this.root = root
  this.parent = parent

  let kExpr = this.io.readU1()
  this.k = kExpr
  let lExpr = this.io.readU1()
  this.l = lExpr
  let mExpr = this.io.readU1()
  this.m = mExpr
  let nExpr = this.io.readU1()
  this.n = nExpr

proc fromFile*(_: typedesc[OpenapiMessage_TimeFamily], filename: string): OpenapiMessage_TimeFamily =
  OpenapiMessage_TimeFamily.read(newKaitaiFileStream(filename), nil, nil)

proc read*(_: typedesc[OpenapiMessage_ValidityFlags], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage_DataQualityBlock): OpenapiMessage_ValidityFlags =
  template this: untyped = result
  this = new(OpenapiMessage_ValidityFlags)
  let root = if root == nil: cast[OpenapiMessage](this) else: cast[OpenapiMessage](root)
  this.io = io
  this.root = root
  this.parent = parent

  let fExpr = this.io.readU2le()
  this.f = fExpr

proc overload(this: OpenapiMessage_ValidityFlags): bool = 
  if this.overloadInst != nil:
    return this.overloadInst
  let overloadInstExpr = bool((this.f and 2) != 0)
  this.overloadInst = overloadInstExpr
  if this.overloadInst != nil:
    return this.overloadInst

proc invalid(this: OpenapiMessage_ValidityFlags): bool = 
  if this.invalidInst != nil:
    return this.invalidInst
  let invalidInstExpr = bool((this.f and 8) != 0)
  this.invalidInst = invalidInstExpr
  if this.invalidInst != nil:
    return this.invalidInst

proc overrun(this: OpenapiMessage_ValidityFlags): bool = 
  if this.overrunInst != nil:
    return this.overrunInst
  let overrunInstExpr = bool((this.f and 16) != 0)
  this.overrunInst = overrunInstExpr
  if this.overrunInst != nil:
    return this.overrunInst

proc fromFile*(_: typedesc[OpenapiMessage_ValidityFlags], filename: string): OpenapiMessage_ValidityFlags =
  OpenapiMessage_ValidityFlags.read(newKaitaiFileStream(filename), nil, nil)

proc read*(_: typedesc[OpenapiMessage_SignalData], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage): OpenapiMessage_SignalData =
  template this: untyped = result
  this = new(OpenapiMessage_SignalData)
  let root = if root == nil: cast[OpenapiMessage](this) else: cast[OpenapiMessage](root)
  this.io = io
  this.root = root
  this.parent = parent

  let numberOfSignalsExpr = this.io.readU2le()
  this.numberOfSignals = numberOfSignalsExpr
  let reservedExpr = this.io.readU2le()
  this.reserved = reservedExpr
  for i in 0 ..< int(this.numberOfSignals):
    let it = OpenapiMessage_SignalBlock.read(this.io, this.root, this)
    this.signals.add(it)

proc fromFile*(_: typedesc[OpenapiMessage_SignalData], filename: string): OpenapiMessage_SignalData =
  OpenapiMessage_SignalData.read(newKaitaiFileStream(filename), nil, nil)

proc read*(_: typedesc[OpenapiMessage_Header], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage): OpenapiMessage_Header =
  template this: untyped = result
  this = new(OpenapiMessage_Header)
  let root = if root == nil: cast[OpenapiMessage](this) else: cast[OpenapiMessage](root)
  this.io = io
  this.root = root
  this.parent = parent

  let magicExpr = this.io.readBytes(int(2))
  this.magic = magicExpr
  let headerLengthExpr = this.io.readU2le()
  this.headerLength = headerLengthExpr
  let messageTypeExpr = OpenapiMessage_Header_EMessageType(this.io.readU2le())
  this.messageType = messageTypeExpr
  let reserved1Expr = this.io.readU2le()
  this.reserved1 = reserved1Expr
  let reserved2Expr = this.io.readU4le()
  this.reserved2 = reserved2Expr
  let timeExpr = OpenapiMessage_Time.read(this.io, this.root, this)
  this.time = timeExpr
  let messageLengthExpr = this.io.readU4le()
  this.messageLength = messageLengthExpr

proc fromFile*(_: typedesc[OpenapiMessage_Header], filename: string): OpenapiMessage_Header =
  OpenapiMessage_Header.read(newKaitaiFileStream(filename), nil, nil)

proc read*(_: typedesc[OpenapiMessage_SignalBlock], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage_SignalData): OpenapiMessage_SignalBlock =
  template this: untyped = result
  this = new(OpenapiMessage_SignalBlock)
  let root = if root == nil: cast[OpenapiMessage](this) else: cast[OpenapiMessage](root)
  this.io = io
  this.root = root
  this.parent = parent

  let signalIdExpr = this.io.readS2le()
  this.signalId = signalIdExpr
  let numberOfValuesExpr = this.io.readS2le()
  this.numberOfValues = numberOfValuesExpr
  for i in 0 ..< int(this.numberOfValues):
    let it = OpenapiMessage_Value.read(this.io, this.root, this)
    this.values.add(it)

proc fromFile*(_: typedesc[OpenapiMessage_SignalBlock], filename: string): OpenapiMessage_SignalBlock =
  OpenapiMessage_SignalBlock.read(newKaitaiFileStream(filename), nil, nil)

proc read*(_: typedesc[OpenapiMessage_Time], io: KaitaiStream, root: KaitaiStruct, parent: KaitaiStruct): OpenapiMessage_Time =
  template this: untyped = result
  this = new(OpenapiMessage_Time)
  let root = if root == nil: cast[OpenapiMessage](this) else: cast[OpenapiMessage](root)
  this.io = io
  this.root = root
  this.parent = parent

  let timeFamilyExpr = OpenapiMessage_TimeFamily.read(this.io, this.root, this)
  this.timeFamily = timeFamilyExpr
  let timeCountExpr = this.io.readU8le()
  this.timeCount = timeCountExpr

proc fromFile*(_: typedesc[OpenapiMessage_Time], filename: string): OpenapiMessage_Time =
  OpenapiMessage_Time.read(newKaitaiFileStream(filename), nil, nil)

proc read*(_: typedesc[OpenapiMessage_Value], io: KaitaiStream, root: KaitaiStruct, parent: OpenapiMessage_SignalBlock): OpenapiMessage_Value =
  template this: untyped = result
  this = new(OpenapiMessage_Value)
  let root = if root == nil: cast[OpenapiMessage](this) else: cast[OpenapiMessage](root)
  this.io = io
  this.root = root
  this.parent = parent

  let value1Expr = this.io.readU1()
  this.value1 = value1Expr
  let value2Expr = this.io.readU1()
  this.value2 = value2Expr
  let value3Expr = this.io.readS1()
  this.value3 = value3Expr

proc calcValue(this: OpenapiMessage_Value): int = 
  if this.calcValueInst != nil:
    return this.calcValueInst
  let calcValueInstExpr = int(((this.value1 + (this.value2 shl 8)) + (this.value3 shl 16)))
  this.calcValueInst = calcValueInstExpr
  if this.calcValueInst != nil:
    return this.calcValueInst

proc fromFile*(_: typedesc[OpenapiMessage_Value], filename: string): OpenapiMessage_Value =
  OpenapiMessage_Value.read(newKaitaiFileStream(filename), nil, nil)

