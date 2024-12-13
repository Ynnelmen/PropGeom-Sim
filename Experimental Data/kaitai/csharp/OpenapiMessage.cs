// This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

using System.Collections.Generic;

namespace Kaitai
{
    public partial class OpenapiMessage : KaitaiStruct
    {
        public static OpenapiMessage FromFile(string fileName)
        {
            return new OpenapiMessage(new KaitaiStream(fileName));
        }

        public OpenapiMessage(KaitaiStream p__io, KaitaiStruct p__parent = null, OpenapiMessage p__root = null) : base(p__io)
        {
            m_parent = p__parent;
            m_root = p__root ?? this;
            _read();
        }
        private void _read()
        {
            _header = new Header(m_io, this, m_root);
            switch (Header.MessageType) {
            case Header.EMessageType.ESignalData: {
                __raw_message = m_io.ReadBytes(Header.MessageLength);
                var io___raw_message = new KaitaiStream(__raw_message);
                _message = new SignalData(io___raw_message, this, m_root);
                break;
            }
            case Header.EMessageType.EDataQuality: {
                __raw_message = m_io.ReadBytes(Header.MessageLength);
                var io___raw_message = new KaitaiStream(__raw_message);
                _message = new DataQuality(io___raw_message, this, m_root);
                break;
            }
            case Header.EMessageType.EInterpretation: {
                __raw_message = m_io.ReadBytes(Header.MessageLength);
                var io___raw_message = new KaitaiStream(__raw_message);
                _message = new Interpretations(io___raw_message, this, m_root);
                break;
            }
            default: {
                _message = m_io.ReadBytes(Header.MessageLength);
                break;
            }
            }
        }
        public partial class Interpretations : KaitaiStruct
        {
            public static Interpretations FromFile(string fileName)
            {
                return new Interpretations(new KaitaiStream(fileName));
            }

            public Interpretations(KaitaiStream p__io, OpenapiMessage p__parent = null, OpenapiMessage p__root = null) : base(p__io)
            {
                m_parent = p__parent;
                m_root = p__root;
                _read();
            }
            private void _read()
            {
                _interpretations = new List<Interpretation>();
                {
                    var i = 0;
                    while (!m_io.IsEof) {
                        _interpretations.Add(new Interpretation(m_io, this, m_root));
                        i++;
                    }
                }
            }
            private List<Interpretation> _interpretations;
            private OpenapiMessage m_root;
            private OpenapiMessage m_parent;
            public List<Interpretation> Interpretations { get { return _interpretations; } }
            public OpenapiMessage M_Root { get { return m_root; } }
            public OpenapiMessage M_Parent { get { return m_parent; } }
        }
        public partial class DataQuality : KaitaiStruct
        {
            public static DataQuality FromFile(string fileName)
            {
                return new DataQuality(new KaitaiStream(fileName));
            }

            public DataQuality(KaitaiStream p__io, OpenapiMessage p__parent = null, OpenapiMessage p__root = null) : base(p__io)
            {
                m_parent = p__parent;
                m_root = p__root;
                _read();
            }
            private void _read()
            {
                _numberOfSignals = m_io.ReadU2le();
                _qualities = new List<DataQualityBlock>((int) (NumberOfSignals));
                for (var i = 0; i < NumberOfSignals; i++)
                {
                    _qualities.Add(new DataQualityBlock(m_io, this, m_root));
                }
            }
            private ushort _numberOfSignals;
            private List<DataQualityBlock> _qualities;
            private OpenapiMessage m_root;
            private OpenapiMessage m_parent;
            public ushort NumberOfSignals { get { return _numberOfSignals; } }
            public List<DataQualityBlock> Qualities { get { return _qualities; } }
            public OpenapiMessage M_Root { get { return m_root; } }
            public OpenapiMessage M_Parent { get { return m_parent; } }
        }
        public partial class DataQualityBlock : KaitaiStruct
        {
            public static DataQualityBlock FromFile(string fileName)
            {
                return new DataQualityBlock(new KaitaiStream(fileName));
            }

            public DataQualityBlock(KaitaiStream p__io, OpenapiMessage.DataQuality p__parent = null, OpenapiMessage p__root = null) : base(p__io)
            {
                m_parent = p__parent;
                m_root = p__root;
                _read();
            }
            private void _read()
            {
                _signalId = m_io.ReadU2le();
                _validityFlags = new ValidityFlags(m_io, this, m_root);
                _reserved = m_io.ReadU2le();
            }
            private ushort _signalId;
            private ValidityFlags _validityFlags;
            private ushort _reserved;
            private OpenapiMessage m_root;
            private OpenapiMessage.DataQuality m_parent;
            public ushort SignalId { get { return _signalId; } }
            public ValidityFlags ValidityFlags { get { return _validityFlags; } }
            public ushort Reserved { get { return _reserved; } }
            public OpenapiMessage M_Root { get { return m_root; } }
            public OpenapiMessage.DataQuality M_Parent { get { return m_parent; } }
        }
        public partial class Interpretation : KaitaiStruct
        {
            public static Interpretation FromFile(string fileName)
            {
                return new Interpretation(new KaitaiStream(fileName));
            }


            public enum EDescriptorType
            {
                DataType = 1,
                ScaleFactor = 2,
                Offset = 3,
                PeriodTime = 4,
                Unit = 5,
                VectorLength = 6,
                ChannelType = 7,
            }
            public Interpretation(KaitaiStream p__io, OpenapiMessage.Interpretations p__parent = null, OpenapiMessage p__root = null) : base(p__io)
            {
                m_parent = p__parent;
                m_root = p__root;
                _read();
            }
            private void _read()
            {
                _signalId = m_io.ReadU2le();
                _descriptorType = ((EDescriptorType) m_io.ReadU2le());
                _reserved = m_io.ReadU2le();
                _valueLength = m_io.ReadU2le();
                switch (DescriptorType) {
                case EDescriptorType.DataType: {
                    _value = m_io.ReadU2le();
                    break;
                }
                case EDescriptorType.ScaleFactor: {
                    _value = m_io.ReadF8le();
                    break;
                }
                case EDescriptorType.Unit: {
                    _value = new String(m_io, this, m_root);
                    break;
                }
                case EDescriptorType.VectorLength: {
                    _value = m_io.ReadU2le();
                    break;
                }
                case EDescriptorType.PeriodTime: {
                    _value = new Time(m_io, this, m_root);
                    break;
                }
                case EDescriptorType.Offset: {
                    _value = m_io.ReadF8le();
                    break;
                }
                case EDescriptorType.ChannelType: {
                    _value = m_io.ReadU2le();
                    break;
                }
                }
                _padding = new List<byte>((int) (((4 - KaitaiStream.Mod(M_Io.Pos, 4)) & 3)));
                for (var i = 0; i < ((4 - KaitaiStream.Mod(M_Io.Pos, 4)) & 3); i++)
                {
                    _padding.Add(m_io.ReadU1());
                }
            }
            private ushort _signalId;
            private EDescriptorType _descriptorType;
            private ushort _reserved;
            private ushort _valueLength;
            private object _value;
            private List<byte> _padding;
            private OpenapiMessage m_root;
            private OpenapiMessage.Interpretations m_parent;
            public ushort SignalId { get { return _signalId; } }
            public EDescriptorType DescriptorType { get { return _descriptorType; } }
            public ushort Reserved { get { return _reserved; } }
            public ushort ValueLength { get { return _valueLength; } }
            public object Value { get { return _value; } }
            public List<byte> Padding { get { return _padding; } }
            public OpenapiMessage M_Root { get { return m_root; } }
            public OpenapiMessage.Interpretations M_Parent { get { return m_parent; } }
        }
        public partial class String : KaitaiStruct
        {
            public static String FromFile(string fileName)
            {
                return new String(new KaitaiStream(fileName));
            }

            public String(KaitaiStream p__io, OpenapiMessage.Interpretation p__parent = null, OpenapiMessage p__root = null) : base(p__io)
            {
                m_parent = p__parent;
                m_root = p__root;
                _read();
            }
            private void _read()
            {
                _count = m_io.ReadU2le();
                _data = System.Text.Encoding.GetEncoding("UTF-8").GetString(m_io.ReadBytes(Count));
            }
            private ushort _count;
            private string _data;
            private OpenapiMessage m_root;
            private OpenapiMessage.Interpretation m_parent;
            public ushort Count { get { return _count; } }
            public string Data { get { return _data; } }
            public OpenapiMessage M_Root { get { return m_root; } }
            public OpenapiMessage.Interpretation M_Parent { get { return m_parent; } }
        }
        public partial class TimeFamily : KaitaiStruct
        {
            public static TimeFamily FromFile(string fileName)
            {
                return new TimeFamily(new KaitaiStream(fileName));
            }

            public TimeFamily(KaitaiStream p__io, OpenapiMessage.Time p__parent = null, OpenapiMessage p__root = null) : base(p__io)
            {
                m_parent = p__parent;
                m_root = p__root;
                _read();
            }
            private void _read()
            {
                _k = m_io.ReadU1();
                _l = m_io.ReadU1();
                _m = m_io.ReadU1();
                _n = m_io.ReadU1();
            }
            private byte _k;
            private byte _l;
            private byte _m;
            private byte _n;
            private OpenapiMessage m_root;
            private OpenapiMessage.Time m_parent;
            public byte K { get { return _k; } }
            public byte L { get { return _l; } }
            public byte M { get { return _m; } }
            public byte N { get { return _n; } }
            public OpenapiMessage M_Root { get { return m_root; } }
            public OpenapiMessage.Time M_Parent { get { return m_parent; } }
        }
        public partial class ValidityFlags : KaitaiStruct
        {
            public static ValidityFlags FromFile(string fileName)
            {
                return new ValidityFlags(new KaitaiStream(fileName));
            }

            public ValidityFlags(KaitaiStream p__io, OpenapiMessage.DataQualityBlock p__parent = null, OpenapiMessage p__root = null) : base(p__io)
            {
                m_parent = p__parent;
                m_root = p__root;
                f_overload = false;
                f_invalid = false;
                f_overrun = false;
                _read();
            }
            private void _read()
            {
                _f = m_io.ReadU2le();
            }
            private bool f_overload;
            private bool _overload;
            public bool Overload
            {
                get
                {
                    if (f_overload)
                        return _overload;
                    _overload = (bool) ((F & 2) != 0);
                    f_overload = true;
                    return _overload;
                }
            }
            private bool f_invalid;
            private bool _invalid;
            public bool Invalid
            {
                get
                {
                    if (f_invalid)
                        return _invalid;
                    _invalid = (bool) ((F & 8) != 0);
                    f_invalid = true;
                    return _invalid;
                }
            }
            private bool f_overrun;
            private bool _overrun;
            public bool Overrun
            {
                get
                {
                    if (f_overrun)
                        return _overrun;
                    _overrun = (bool) ((F & 16) != 0);
                    f_overrun = true;
                    return _overrun;
                }
            }
            private ushort _f;
            private OpenapiMessage m_root;
            private OpenapiMessage.DataQualityBlock m_parent;
            public ushort F { get { return _f; } }
            public OpenapiMessage M_Root { get { return m_root; } }
            public OpenapiMessage.DataQualityBlock M_Parent { get { return m_parent; } }
        }
        public partial class SignalData : KaitaiStruct
        {
            public static SignalData FromFile(string fileName)
            {
                return new SignalData(new KaitaiStream(fileName));
            }

            public SignalData(KaitaiStream p__io, OpenapiMessage p__parent = null, OpenapiMessage p__root = null) : base(p__io)
            {
                m_parent = p__parent;
                m_root = p__root;
                _read();
            }
            private void _read()
            {
                _numberOfSignals = m_io.ReadU2le();
                _reserved = m_io.ReadU2le();
                _signals = new List<SignalBlock>((int) (NumberOfSignals));
                for (var i = 0; i < NumberOfSignals; i++)
                {
                    _signals.Add(new SignalBlock(m_io, this, m_root));
                }
            }
            private ushort _numberOfSignals;
            private ushort _reserved;
            private List<SignalBlock> _signals;
            private OpenapiMessage m_root;
            private OpenapiMessage m_parent;
            public ushort NumberOfSignals { get { return _numberOfSignals; } }
            public ushort Reserved { get { return _reserved; } }
            public List<SignalBlock> Signals { get { return _signals; } }
            public OpenapiMessage M_Root { get { return m_root; } }
            public OpenapiMessage M_Parent { get { return m_parent; } }
        }
        public partial class Header : KaitaiStruct
        {
            public static Header FromFile(string fileName)
            {
                return new Header(new KaitaiStream(fileName));
            }


            public enum EMessageType
            {
                ESignalData = 1,
                EDataQuality = 2,
                EInterpretation = 8,
                EAuxSequenceData = 11,
            }
            public Header(KaitaiStream p__io, OpenapiMessage p__parent = null, OpenapiMessage p__root = null) : base(p__io)
            {
                m_parent = p__parent;
                m_root = p__root;
                _read();
            }
            private void _read()
            {
                _magic = m_io.ReadBytes(2);
                if (!((KaitaiStream.ByteArrayCompare(Magic, new byte[] { 66, 75 }) == 0)))
                {
                    throw new ValidationNotEqualError(new byte[] { 66, 75 }, Magic, M_Io, "/types/header/seq/0");
                }
                _headerLength = m_io.ReadU2le();
                _messageType = ((EMessageType) m_io.ReadU2le());
                _reserved1 = m_io.ReadU2le();
                _reserved2 = m_io.ReadU4le();
                _time = new Time(m_io, this, m_root);
                _messageLength = m_io.ReadU4le();
            }
            private byte[] _magic;
            private ushort _headerLength;
            private EMessageType _messageType;
            private ushort _reserved1;
            private uint _reserved2;
            private Time _time;
            private uint _messageLength;
            private OpenapiMessage m_root;
            private OpenapiMessage m_parent;
            public byte[] Magic { get { return _magic; } }
            public ushort HeaderLength { get { return _headerLength; } }
            public EMessageType MessageType { get { return _messageType; } }
            public ushort Reserved1 { get { return _reserved1; } }
            public uint Reserved2 { get { return _reserved2; } }
            public Time Time { get { return _time; } }
            public uint MessageLength { get { return _messageLength; } }
            public OpenapiMessage M_Root { get { return m_root; } }
            public OpenapiMessage M_Parent { get { return m_parent; } }
        }
        public partial class SignalBlock : KaitaiStruct
        {
            public static SignalBlock FromFile(string fileName)
            {
                return new SignalBlock(new KaitaiStream(fileName));
            }

            public SignalBlock(KaitaiStream p__io, OpenapiMessage.SignalData p__parent = null, OpenapiMessage p__root = null) : base(p__io)
            {
                m_parent = p__parent;
                m_root = p__root;
                _read();
            }
            private void _read()
            {
                _signalId = m_io.ReadS2le();
                _numberOfValues = m_io.ReadS2le();
                _values = new List<Value>((int) (NumberOfValues));
                for (var i = 0; i < NumberOfValues; i++)
                {
                    _values.Add(new Value(m_io, this, m_root));
                }
            }
            private short _signalId;
            private short _numberOfValues;
            private List<Value> _values;
            private OpenapiMessage m_root;
            private OpenapiMessage.SignalData m_parent;
            public short SignalId { get { return _signalId; } }
            public short NumberOfValues { get { return _numberOfValues; } }
            public List<Value> Values { get { return _values; } }
            public OpenapiMessage M_Root { get { return m_root; } }
            public OpenapiMessage.SignalData M_Parent { get { return m_parent; } }
        }
        public partial class Time : KaitaiStruct
        {
            public static Time FromFile(string fileName)
            {
                return new Time(new KaitaiStream(fileName));
            }

            public Time(KaitaiStream p__io, KaitaiStruct p__parent = null, OpenapiMessage p__root = null) : base(p__io)
            {
                m_parent = p__parent;
                m_root = p__root;
                _read();
            }
            private void _read()
            {
                _timeFamily = new TimeFamily(m_io, this, m_root);
                _timeCount = m_io.ReadU8le();
            }
            private TimeFamily _timeFamily;
            private ulong _timeCount;
            private OpenapiMessage m_root;
            private KaitaiStruct m_parent;
            public TimeFamily TimeFamily { get { return _timeFamily; } }
            public ulong TimeCount { get { return _timeCount; } }
            public OpenapiMessage M_Root { get { return m_root; } }
            public KaitaiStruct M_Parent { get { return m_parent; } }
        }
        public partial class Value : KaitaiStruct
        {
            public static Value FromFile(string fileName)
            {
                return new Value(new KaitaiStream(fileName));
            }

            public Value(KaitaiStream p__io, OpenapiMessage.SignalBlock p__parent = null, OpenapiMessage p__root = null) : base(p__io)
            {
                m_parent = p__parent;
                m_root = p__root;
                f_calcValue = false;
                _read();
            }
            private void _read()
            {
                _value1 = m_io.ReadU1();
                _value2 = m_io.ReadU1();
                _value3 = m_io.ReadS1();
            }
            private bool f_calcValue;
            private int _calcValue;
            public int CalcValue
            {
                get
                {
                    if (f_calcValue)
                        return _calcValue;
                    _calcValue = (int) (((Value1 + (Value2 << 8)) + (Value3 << 16)));
                    f_calcValue = true;
                    return _calcValue;
                }
            }
            private byte _value1;
            private byte _value2;
            private sbyte _value3;
            private OpenapiMessage m_root;
            private OpenapiMessage.SignalBlock m_parent;
            public byte Value1 { get { return _value1; } }
            public byte Value2 { get { return _value2; } }
            public sbyte Value3 { get { return _value3; } }
            public OpenapiMessage M_Root { get { return m_root; } }
            public OpenapiMessage.SignalBlock M_Parent { get { return m_parent; } }
        }
        private Header _header;
        private object _message;
        private OpenapiMessage m_root;
        private KaitaiStruct m_parent;
        private byte[] __raw_message;
        public Header Header { get { return _header; } }
        public object Message { get { return _message; } }
        public OpenapiMessage M_Root { get { return m_root; } }
        public KaitaiStruct M_Parent { get { return m_parent; } }
        public byte[] M_RawMessage { get { return __raw_message; } }
    }
}
