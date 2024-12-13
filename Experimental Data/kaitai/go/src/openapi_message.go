// This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

import (
	"github.com/kaitai-io/kaitai_struct_go_runtime/kaitai"
	"bytes"
)

type OpenapiMessage struct {
	Header *OpenapiMessage_Header
	Message interface{}
	_io *kaitai.Stream
	_root *OpenapiMessage
	_parent interface{}
	_raw_Message []byte
}
func NewOpenapiMessage() *OpenapiMessage {
	return &OpenapiMessage{
	}
}

func (this *OpenapiMessage) Read(io *kaitai.Stream, parent interface{}, root *OpenapiMessage) (err error) {
	this._io = io
	this._parent = parent
	this._root = root

	tmp1 := NewOpenapiMessage_Header()
	err = tmp1.Read(this._io, this, this._root)
	if err != nil {
		return err
	}
	this.Header = tmp1
	switch (this.Header.MessageType) {
	case OpenapiMessage_Header_EMessageType__ESignalData:
		tmp2, err := this._io.ReadBytes(int(this.Header.MessageLength))
		if err != nil {
			return err
		}
		tmp2 = tmp2
		this._raw_Message = tmp2
		_io__raw_Message := kaitai.NewStream(bytes.NewReader(this._raw_Message))
		tmp3 := NewOpenapiMessage_SignalData()
		err = tmp3.Read(_io__raw_Message, this, this._root)
		if err != nil {
			return err
		}
		this.Message = tmp3
	case OpenapiMessage_Header_EMessageType__EDataQuality:
		tmp4, err := this._io.ReadBytes(int(this.Header.MessageLength))
		if err != nil {
			return err
		}
		tmp4 = tmp4
		this._raw_Message = tmp4
		_io__raw_Message := kaitai.NewStream(bytes.NewReader(this._raw_Message))
		tmp5 := NewOpenapiMessage_DataQuality()
		err = tmp5.Read(_io__raw_Message, this, this._root)
		if err != nil {
			return err
		}
		this.Message = tmp5
	case OpenapiMessage_Header_EMessageType__EInterpretation:
		tmp6, err := this._io.ReadBytes(int(this.Header.MessageLength))
		if err != nil {
			return err
		}
		tmp6 = tmp6
		this._raw_Message = tmp6
		_io__raw_Message := kaitai.NewStream(bytes.NewReader(this._raw_Message))
		tmp7 := NewOpenapiMessage_Interpretations()
		err = tmp7.Read(_io__raw_Message, this, this._root)
		if err != nil {
			return err
		}
		this.Message = tmp7
	default:
		tmp8, err := this._io.ReadBytes(int(this.Header.MessageLength))
		if err != nil {
			return err
		}
		tmp8 = tmp8
		this._raw_Message = tmp8
	}
	return err
}
type OpenapiMessage_Interpretations struct {
	Interpretations []*OpenapiMessage_Interpretation
	_io *kaitai.Stream
	_root *OpenapiMessage
	_parent *OpenapiMessage
}
func NewOpenapiMessage_Interpretations() *OpenapiMessage_Interpretations {
	return &OpenapiMessage_Interpretations{
	}
}

func (this *OpenapiMessage_Interpretations) Read(io *kaitai.Stream, parent *OpenapiMessage, root *OpenapiMessage) (err error) {
	this._io = io
	this._parent = parent
	this._root = root

	for i := 1;; i++ {
		tmp9, err := this._io.EOF()
		if err != nil {
			return err
		}
		if tmp9 {
			break
		}
		tmp10 := NewOpenapiMessage_Interpretation()
		err = tmp10.Read(this._io, this, this._root)
		if err != nil {
			return err
		}
		this.Interpretations = append(this.Interpretations, tmp10)
	}
	return err
}
type OpenapiMessage_DataQuality struct {
	NumberOfSignals uint16
	Qualities []*OpenapiMessage_DataQualityBlock
	_io *kaitai.Stream
	_root *OpenapiMessage
	_parent *OpenapiMessage
}
func NewOpenapiMessage_DataQuality() *OpenapiMessage_DataQuality {
	return &OpenapiMessage_DataQuality{
	}
}

func (this *OpenapiMessage_DataQuality) Read(io *kaitai.Stream, parent *OpenapiMessage, root *OpenapiMessage) (err error) {
	this._io = io
	this._parent = parent
	this._root = root

	tmp11, err := this._io.ReadU2le()
	if err != nil {
		return err
	}
	this.NumberOfSignals = uint16(tmp11)
	this.Qualities = make([]*OpenapiMessage_DataQualityBlock, this.NumberOfSignals)
	for i := range this.Qualities {
		tmp12 := NewOpenapiMessage_DataQualityBlock()
		err = tmp12.Read(this._io, this, this._root)
		if err != nil {
			return err
		}
		this.Qualities[i] = tmp12
	}
	return err
}
type OpenapiMessage_DataQualityBlock struct {
	SignalId uint16
	ValidityFlags *OpenapiMessage_ValidityFlags
	Reserved uint16
	_io *kaitai.Stream
	_root *OpenapiMessage
	_parent *OpenapiMessage_DataQuality
}
func NewOpenapiMessage_DataQualityBlock() *OpenapiMessage_DataQualityBlock {
	return &OpenapiMessage_DataQualityBlock{
	}
}

func (this *OpenapiMessage_DataQualityBlock) Read(io *kaitai.Stream, parent *OpenapiMessage_DataQuality, root *OpenapiMessage) (err error) {
	this._io = io
	this._parent = parent
	this._root = root

	tmp13, err := this._io.ReadU2le()
	if err != nil {
		return err
	}
	this.SignalId = uint16(tmp13)
	tmp14 := NewOpenapiMessage_ValidityFlags()
	err = tmp14.Read(this._io, this, this._root)
	if err != nil {
		return err
	}
	this.ValidityFlags = tmp14
	tmp15, err := this._io.ReadU2le()
	if err != nil {
		return err
	}
	this.Reserved = uint16(tmp15)
	return err
}

type OpenapiMessage_Interpretation_EDescriptorType int
const (
	OpenapiMessage_Interpretation_EDescriptorType__DataType OpenapiMessage_Interpretation_EDescriptorType = 1
	OpenapiMessage_Interpretation_EDescriptorType__ScaleFactor OpenapiMessage_Interpretation_EDescriptorType = 2
	OpenapiMessage_Interpretation_EDescriptorType__Offset OpenapiMessage_Interpretation_EDescriptorType = 3
	OpenapiMessage_Interpretation_EDescriptorType__PeriodTime OpenapiMessage_Interpretation_EDescriptorType = 4
	OpenapiMessage_Interpretation_EDescriptorType__Unit OpenapiMessage_Interpretation_EDescriptorType = 5
	OpenapiMessage_Interpretation_EDescriptorType__VectorLength OpenapiMessage_Interpretation_EDescriptorType = 6
	OpenapiMessage_Interpretation_EDescriptorType__ChannelType OpenapiMessage_Interpretation_EDescriptorType = 7
)
type OpenapiMessage_Interpretation struct {
	SignalId uint16
	DescriptorType OpenapiMessage_Interpretation_EDescriptorType
	Reserved uint16
	ValueLength uint16
	Value interface{}
	Padding []uint8
	_io *kaitai.Stream
	_root *OpenapiMessage
	_parent *OpenapiMessage_Interpretations
}
func NewOpenapiMessage_Interpretation() *OpenapiMessage_Interpretation {
	return &OpenapiMessage_Interpretation{
	}
}

func (this *OpenapiMessage_Interpretation) Read(io *kaitai.Stream, parent *OpenapiMessage_Interpretations, root *OpenapiMessage) (err error) {
	this._io = io
	this._parent = parent
	this._root = root

	tmp16, err := this._io.ReadU2le()
	if err != nil {
		return err
	}
	this.SignalId = uint16(tmp16)
	tmp17, err := this._io.ReadU2le()
	if err != nil {
		return err
	}
	this.DescriptorType = OpenapiMessage_Interpretation_EDescriptorType(tmp17)
	tmp18, err := this._io.ReadU2le()
	if err != nil {
		return err
	}
	this.Reserved = uint16(tmp18)
	tmp19, err := this._io.ReadU2le()
	if err != nil {
		return err
	}
	this.ValueLength = uint16(tmp19)
	switch (this.DescriptorType) {
	case OpenapiMessage_Interpretation_EDescriptorType__DataType:
		tmp20, err := this._io.ReadU2le()
		if err != nil {
			return err
		}
		this.Value = tmp20
	case OpenapiMessage_Interpretation_EDescriptorType__ScaleFactor:
		tmp21, err := this._io.ReadF8le()
		if err != nil {
			return err
		}
		this.Value = tmp21
	case OpenapiMessage_Interpretation_EDescriptorType__Unit:
		tmp22 := NewOpenapiMessage_String()
		err = tmp22.Read(this._io, this, this._root)
		if err != nil {
			return err
		}
		this.Value = tmp22
	case OpenapiMessage_Interpretation_EDescriptorType__VectorLength:
		tmp23, err := this._io.ReadU2le()
		if err != nil {
			return err
		}
		this.Value = tmp23
	case OpenapiMessage_Interpretation_EDescriptorType__PeriodTime:
		tmp24 := NewOpenapiMessage_Time()
		err = tmp24.Read(this._io, this, this._root)
		if err != nil {
			return err
		}
		this.Value = tmp24
	case OpenapiMessage_Interpretation_EDescriptorType__Offset:
		tmp25, err := this._io.ReadF8le()
		if err != nil {
			return err
		}
		this.Value = tmp25
	case OpenapiMessage_Interpretation_EDescriptorType__ChannelType:
		tmp26, err := this._io.ReadU2le()
		if err != nil {
			return err
		}
		this.Value = tmp26
	}
	tmp28, err := this._io.Pos()
	if err != nil {
		return err
	}
	tmp27 := tmp28 % 4
	if tmp27 < 0 {
		tmp27 += 4
	}
	this.Padding = make([]uint8, ((4 - tmp27) & 3))
	for i := range this.Padding {
		tmp29, err := this._io.ReadU1()
		if err != nil {
			return err
		}
		this.Padding[i] = tmp29
	}
	return err
}
type OpenapiMessage_String struct {
	Count uint16
	Data string
	_io *kaitai.Stream
	_root *OpenapiMessage
	_parent *OpenapiMessage_Interpretation
}
func NewOpenapiMessage_String() *OpenapiMessage_String {
	return &OpenapiMessage_String{
	}
}

func (this *OpenapiMessage_String) Read(io *kaitai.Stream, parent *OpenapiMessage_Interpretation, root *OpenapiMessage) (err error) {
	this._io = io
	this._parent = parent
	this._root = root

	tmp30, err := this._io.ReadU2le()
	if err != nil {
		return err
	}
	this.Count = uint16(tmp30)
	tmp31, err := this._io.ReadBytes(int(this.Count))
	if err != nil {
		return err
	}
	tmp31 = tmp31
	this.Data = string(tmp31)
	return err
}
type OpenapiMessage_TimeFamily struct {
	K uint8
	L uint8
	M uint8
	N uint8
	_io *kaitai.Stream
	_root *OpenapiMessage
	_parent *OpenapiMessage_Time
}
func NewOpenapiMessage_TimeFamily() *OpenapiMessage_TimeFamily {
	return &OpenapiMessage_TimeFamily{
	}
}

func (this *OpenapiMessage_TimeFamily) Read(io *kaitai.Stream, parent *OpenapiMessage_Time, root *OpenapiMessage) (err error) {
	this._io = io
	this._parent = parent
	this._root = root

	tmp32, err := this._io.ReadU1()
	if err != nil {
		return err
	}
	this.K = tmp32
	tmp33, err := this._io.ReadU1()
	if err != nil {
		return err
	}
	this.L = tmp33
	tmp34, err := this._io.ReadU1()
	if err != nil {
		return err
	}
	this.M = tmp34
	tmp35, err := this._io.ReadU1()
	if err != nil {
		return err
	}
	this.N = tmp35
	return err
}
type OpenapiMessage_ValidityFlags struct {
	F uint16
	_io *kaitai.Stream
	_root *OpenapiMessage
	_parent *OpenapiMessage_DataQualityBlock
	_f_overload bool
	overload bool
	_f_invalid bool
	invalid bool
	_f_overrun bool
	overrun bool
}
func NewOpenapiMessage_ValidityFlags() *OpenapiMessage_ValidityFlags {
	return &OpenapiMessage_ValidityFlags{
	}
}

func (this *OpenapiMessage_ValidityFlags) Read(io *kaitai.Stream, parent *OpenapiMessage_DataQualityBlock, root *OpenapiMessage) (err error) {
	this._io = io
	this._parent = parent
	this._root = root

	tmp36, err := this._io.ReadU2le()
	if err != nil {
		return err
	}
	this.F = uint16(tmp36)
	return err
}
func (this *OpenapiMessage_ValidityFlags) Overload() (v bool, err error) {
	if (this._f_overload) {
		return this.overload, nil
	}
	this.overload = bool((this.F & 2) != 0)
	this._f_overload = true
	return this.overload, nil
}
func (this *OpenapiMessage_ValidityFlags) Invalid() (v bool, err error) {
	if (this._f_invalid) {
		return this.invalid, nil
	}
	this.invalid = bool((this.F & 8) != 0)
	this._f_invalid = true
	return this.invalid, nil
}
func (this *OpenapiMessage_ValidityFlags) Overrun() (v bool, err error) {
	if (this._f_overrun) {
		return this.overrun, nil
	}
	this.overrun = bool((this.F & 16) != 0)
	this._f_overrun = true
	return this.overrun, nil
}
type OpenapiMessage_SignalData struct {
	NumberOfSignals uint16
	Reserved uint16
	Signals []*OpenapiMessage_SignalBlock
	_io *kaitai.Stream
	_root *OpenapiMessage
	_parent *OpenapiMessage
}
func NewOpenapiMessage_SignalData() *OpenapiMessage_SignalData {
	return &OpenapiMessage_SignalData{
	}
}

func (this *OpenapiMessage_SignalData) Read(io *kaitai.Stream, parent *OpenapiMessage, root *OpenapiMessage) (err error) {
	this._io = io
	this._parent = parent
	this._root = root

	tmp37, err := this._io.ReadU2le()
	if err != nil {
		return err
	}
	this.NumberOfSignals = uint16(tmp37)
	tmp38, err := this._io.ReadU2le()
	if err != nil {
		return err
	}
	this.Reserved = uint16(tmp38)
	this.Signals = make([]*OpenapiMessage_SignalBlock, this.NumberOfSignals)
	for i := range this.Signals {
		tmp39 := NewOpenapiMessage_SignalBlock()
		err = tmp39.Read(this._io, this, this._root)
		if err != nil {
			return err
		}
		this.Signals[i] = tmp39
	}
	return err
}

type OpenapiMessage_Header_EMessageType int
const (
	OpenapiMessage_Header_EMessageType__ESignalData OpenapiMessage_Header_EMessageType = 1
	OpenapiMessage_Header_EMessageType__EDataQuality OpenapiMessage_Header_EMessageType = 2
	OpenapiMessage_Header_EMessageType__EInterpretation OpenapiMessage_Header_EMessageType = 8
	OpenapiMessage_Header_EMessageType__EAuxSequenceData OpenapiMessage_Header_EMessageType = 11
)
type OpenapiMessage_Header struct {
	Magic []byte
	HeaderLength uint16
	MessageType OpenapiMessage_Header_EMessageType
	Reserved1 uint16
	Reserved2 uint32
	Time *OpenapiMessage_Time
	MessageLength uint32
	_io *kaitai.Stream
	_root *OpenapiMessage
	_parent *OpenapiMessage
}
func NewOpenapiMessage_Header() *OpenapiMessage_Header {
	return &OpenapiMessage_Header{
	}
}

func (this *OpenapiMessage_Header) Read(io *kaitai.Stream, parent *OpenapiMessage, root *OpenapiMessage) (err error) {
	this._io = io
	this._parent = parent
	this._root = root

	tmp40, err := this._io.ReadBytes(int(2))
	if err != nil {
		return err
	}
	tmp40 = tmp40
	this.Magic = tmp40
	if !(bytes.Equal(this.Magic, []uint8{66, 75})) {
		return kaitai.NewValidationNotEqualError([]uint8{66, 75}, this.Magic, this._io, "/types/header/seq/0")
	}
	tmp41, err := this._io.ReadU2le()
	if err != nil {
		return err
	}
	this.HeaderLength = uint16(tmp41)
	tmp42, err := this._io.ReadU2le()
	if err != nil {
		return err
	}
	this.MessageType = OpenapiMessage_Header_EMessageType(tmp42)
	tmp43, err := this._io.ReadU2le()
	if err != nil {
		return err
	}
	this.Reserved1 = uint16(tmp43)
	tmp44, err := this._io.ReadU4le()
	if err != nil {
		return err
	}
	this.Reserved2 = uint32(tmp44)
	tmp45 := NewOpenapiMessage_Time()
	err = tmp45.Read(this._io, this, this._root)
	if err != nil {
		return err
	}
	this.Time = tmp45
	tmp46, err := this._io.ReadU4le()
	if err != nil {
		return err
	}
	this.MessageLength = uint32(tmp46)
	return err
}
type OpenapiMessage_SignalBlock struct {
	SignalId int16
	NumberOfValues int16
	Values []*OpenapiMessage_Value
	_io *kaitai.Stream
	_root *OpenapiMessage
	_parent *OpenapiMessage_SignalData
}
func NewOpenapiMessage_SignalBlock() *OpenapiMessage_SignalBlock {
	return &OpenapiMessage_SignalBlock{
	}
}

func (this *OpenapiMessage_SignalBlock) Read(io *kaitai.Stream, parent *OpenapiMessage_SignalData, root *OpenapiMessage) (err error) {
	this._io = io
	this._parent = parent
	this._root = root

	tmp47, err := this._io.ReadS2le()
	if err != nil {
		return err
	}
	this.SignalId = int16(tmp47)
	tmp48, err := this._io.ReadS2le()
	if err != nil {
		return err
	}
	this.NumberOfValues = int16(tmp48)
	this.Values = make([]*OpenapiMessage_Value, this.NumberOfValues)
	for i := range this.Values {
		tmp49 := NewOpenapiMessage_Value()
		err = tmp49.Read(this._io, this, this._root)
		if err != nil {
			return err
		}
		this.Values[i] = tmp49
	}
	return err
}
type OpenapiMessage_Time struct {
	TimeFamily *OpenapiMessage_TimeFamily
	TimeCount uint64
	_io *kaitai.Stream
	_root *OpenapiMessage
	_parent interface{}
}
func NewOpenapiMessage_Time() *OpenapiMessage_Time {
	return &OpenapiMessage_Time{
	}
}

func (this *OpenapiMessage_Time) Read(io *kaitai.Stream, parent interface{}, root *OpenapiMessage) (err error) {
	this._io = io
	this._parent = parent
	this._root = root

	tmp50 := NewOpenapiMessage_TimeFamily()
	err = tmp50.Read(this._io, this, this._root)
	if err != nil {
		return err
	}
	this.TimeFamily = tmp50
	tmp51, err := this._io.ReadU8le()
	if err != nil {
		return err
	}
	this.TimeCount = uint64(tmp51)
	return err
}
type OpenapiMessage_Value struct {
	Value1 uint8
	Value2 uint8
	Value3 int8
	_io *kaitai.Stream
	_root *OpenapiMessage
	_parent *OpenapiMessage_SignalBlock
	_f_calcValue bool
	calcValue int
}
func NewOpenapiMessage_Value() *OpenapiMessage_Value {
	return &OpenapiMessage_Value{
	}
}

func (this *OpenapiMessage_Value) Read(io *kaitai.Stream, parent *OpenapiMessage_SignalBlock, root *OpenapiMessage) (err error) {
	this._io = io
	this._parent = parent
	this._root = root

	tmp52, err := this._io.ReadU1()
	if err != nil {
		return err
	}
	this.Value1 = tmp52
	tmp53, err := this._io.ReadU1()
	if err != nil {
		return err
	}
	this.Value2 = tmp53
	tmp54, err := this._io.ReadS1()
	if err != nil {
		return err
	}
	this.Value3 = tmp54
	return err
}
func (this *OpenapiMessage_Value) CalcValue() (v int, err error) {
	if (this._f_calcValue) {
		return this.calcValue, nil
	}
	this.calcValue = int(((this.Value1 + (this.Value2 << 8)) + (this.Value3 << 16)))
	this._f_calcValue = true
	return this.calcValue, nil
}
