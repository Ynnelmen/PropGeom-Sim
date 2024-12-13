// This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

import io.kaitai.struct.ByteBufferKaitaiStream;
import io.kaitai.struct.KaitaiStruct;
import io.kaitai.struct.KaitaiStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.nio.charset.Charset;
import java.util.Arrays;

public class OpenapiMessage extends KaitaiStruct {
    public static OpenapiMessage fromFile(String fileName) throws IOException {
        return new OpenapiMessage(new ByteBufferKaitaiStream(fileName));
    }

    public OpenapiMessage(KaitaiStream _io) {
        this(_io, null, null);
    }

    public OpenapiMessage(KaitaiStream _io, KaitaiStruct _parent) {
        this(_io, _parent, null);
    }

    public OpenapiMessage(KaitaiStream _io, KaitaiStruct _parent, OpenapiMessage _root) {
        super(_io);
        this._parent = _parent;
        this._root = _root == null ? this : _root;
        _read();
    }
    private void _read() {
        this.header = new Header(this._io, this, _root);
        {
            EMessageType on = header().messageType();
            if (on != null) {
                switch (header().messageType()) {
                case E_SIGNAL_DATA: {
                    this._raw_message = this._io.readBytes(header().messageLength());
                    KaitaiStream _io__raw_message = new ByteBufferKaitaiStream(_raw_message);
                    this.message = new SignalData(_io__raw_message, this, _root);
                    break;
                }
                case E_DATA_QUALITY: {
                    this._raw_message = this._io.readBytes(header().messageLength());
                    KaitaiStream _io__raw_message = new ByteBufferKaitaiStream(_raw_message);
                    this.message = new DataQuality(_io__raw_message, this, _root);
                    break;
                }
                case E_INTERPRETATION: {
                    this._raw_message = this._io.readBytes(header().messageLength());
                    KaitaiStream _io__raw_message = new ByteBufferKaitaiStream(_raw_message);
                    this.message = new Interpretations(_io__raw_message, this, _root);
                    break;
                }
                default: {
                    this.message = this._io.readBytes(header().messageLength());
                    break;
                }
                }
            } else {
                this.message = this._io.readBytes(header().messageLength());
            }
        }
    }
    public static class Interpretations extends KaitaiStruct {
        public static Interpretations fromFile(String fileName) throws IOException {
            return new Interpretations(new ByteBufferKaitaiStream(fileName));
        }

        public Interpretations(KaitaiStream _io) {
            this(_io, null, null);
        }

        public Interpretations(KaitaiStream _io, OpenapiMessage _parent) {
            this(_io, _parent, null);
        }

        public Interpretations(KaitaiStream _io, OpenapiMessage _parent, OpenapiMessage _root) {
            super(_io);
            this._parent = _parent;
            this._root = _root;
            _read();
        }
        private void _read() {
            this.interpretations = new ArrayList<Interpretation>();
            {
                int i = 0;
                while (!this._io.isEof()) {
                    this.interpretations.add(new Interpretation(this._io, this, _root));
                    i++;
                }
            }
        }
        private ArrayList<Interpretation> interpretations;
        private OpenapiMessage _root;
        private OpenapiMessage _parent;
        public ArrayList<Interpretation> interpretations() { return interpretations; }
        public OpenapiMessage _root() { return _root; }
        public OpenapiMessage _parent() { return _parent; }
    }
    public static class DataQuality extends KaitaiStruct {
        public static DataQuality fromFile(String fileName) throws IOException {
            return new DataQuality(new ByteBufferKaitaiStream(fileName));
        }

        public DataQuality(KaitaiStream _io) {
            this(_io, null, null);
        }

        public DataQuality(KaitaiStream _io, OpenapiMessage _parent) {
            this(_io, _parent, null);
        }

        public DataQuality(KaitaiStream _io, OpenapiMessage _parent, OpenapiMessage _root) {
            super(_io);
            this._parent = _parent;
            this._root = _root;
            _read();
        }
        private void _read() {
            this.numberOfSignals = this._io.readU2le();
            qualities = new ArrayList<DataQualityBlock>(((Number) (numberOfSignals())).intValue());
            for (int i = 0; i < numberOfSignals(); i++) {
                this.qualities.add(new DataQualityBlock(this._io, this, _root));
            }
        }
        private int numberOfSignals;
        private ArrayList<DataQualityBlock> qualities;
        private OpenapiMessage _root;
        private OpenapiMessage _parent;
        public int numberOfSignals() { return numberOfSignals; }
        public ArrayList<DataQualityBlock> qualities() { return qualities; }
        public OpenapiMessage _root() { return _root; }
        public OpenapiMessage _parent() { return _parent; }
    }
    public static class DataQualityBlock extends KaitaiStruct {
        public static DataQualityBlock fromFile(String fileName) throws IOException {
            return new DataQualityBlock(new ByteBufferKaitaiStream(fileName));
        }

        public DataQualityBlock(KaitaiStream _io) {
            this(_io, null, null);
        }

        public DataQualityBlock(KaitaiStream _io, OpenapiMessage.DataQuality _parent) {
            this(_io, _parent, null);
        }

        public DataQualityBlock(KaitaiStream _io, OpenapiMessage.DataQuality _parent, OpenapiMessage _root) {
            super(_io);
            this._parent = _parent;
            this._root = _root;
            _read();
        }
        private void _read() {
            this.signalId = this._io.readU2le();
            this.validityFlags = new ValidityFlags(this._io, this, _root);
            this.reserved = this._io.readU2le();
        }
        private int signalId;
        private ValidityFlags validityFlags;
        private int reserved;
        private OpenapiMessage _root;
        private OpenapiMessage.DataQuality _parent;
        public int signalId() { return signalId; }
        public ValidityFlags validityFlags() { return validityFlags; }
        public int reserved() { return reserved; }
        public OpenapiMessage _root() { return _root; }
        public OpenapiMessage.DataQuality _parent() { return _parent; }
    }
    public static class Interpretation extends KaitaiStruct {
        public static Interpretation fromFile(String fileName) throws IOException {
            return new Interpretation(new ByteBufferKaitaiStream(fileName));
        }

        public enum EDescriptorType {
            DATA_TYPE(1),
            SCALE_FACTOR(2),
            OFFSET(3),
            PERIOD_TIME(4),
            UNIT(5),
            VECTOR_LENGTH(6),
            CHANNEL_TYPE(7);

            private final long id;
            EDescriptorType(long id) { this.id = id; }
            public long id() { return id; }
            private static final Map<Long, EDescriptorType> byId = new HashMap<Long, EDescriptorType>(7);
            static {
                for (EDescriptorType e : EDescriptorType.values())
                    byId.put(e.id(), e);
            }
            public static EDescriptorType byId(long id) { return byId.get(id); }
        }

        public Interpretation(KaitaiStream _io) {
            this(_io, null, null);
        }

        public Interpretation(KaitaiStream _io, OpenapiMessage.Interpretations _parent) {
            this(_io, _parent, null);
        }

        public Interpretation(KaitaiStream _io, OpenapiMessage.Interpretations _parent, OpenapiMessage _root) {
            super(_io);
            this._parent = _parent;
            this._root = _root;
            _read();
        }
        private void _read() {
            this.signalId = this._io.readU2le();
            this.descriptorType = EDescriptorType.byId(this._io.readU2le());
            this.reserved = this._io.readU2le();
            this.valueLength = this._io.readU2le();
            {
                EDescriptorType on = descriptorType();
                if (on != null) {
                    switch (descriptorType()) {
                    case DATA_TYPE: {
                        this.value = (Object) (this._io.readU2le());
                        break;
                    }
                    case SCALE_FACTOR: {
                        this.value = (Object) (this._io.readF8le());
                        break;
                    }
                    case UNIT: {
                        this.value = new String(this._io, this, _root);
                        break;
                    }
                    case VECTOR_LENGTH: {
                        this.value = (Object) (this._io.readU2le());
                        break;
                    }
                    case PERIOD_TIME: {
                        this.value = new Time(this._io, this, _root);
                        break;
                    }
                    case OFFSET: {
                        this.value = (Object) (this._io.readF8le());
                        break;
                    }
                    case CHANNEL_TYPE: {
                        this.value = (Object) (this._io.readU2le());
                        break;
                    }
                    }
                }
            }
            padding = new ArrayList<Integer>(((Number) (((4 - KaitaiStream.mod(_io().pos(), 4)) & 3))).intValue());
            for (int i = 0; i < ((4 - KaitaiStream.mod(_io().pos(), 4)) & 3); i++) {
                this.padding.add(this._io.readU1());
            }
        }
        private int signalId;
        private EDescriptorType descriptorType;
        private int reserved;
        private int valueLength;
        private Object value;
        private ArrayList<Integer> padding;
        private OpenapiMessage _root;
        private OpenapiMessage.Interpretations _parent;
        public int signalId() { return signalId; }
        public EDescriptorType descriptorType() { return descriptorType; }
        public int reserved() { return reserved; }
        public int valueLength() { return valueLength; }
        public Object value() { return value; }
        public ArrayList<Integer> padding() { return padding; }
        public OpenapiMessage _root() { return _root; }
        public OpenapiMessage.Interpretations _parent() { return _parent; }
    }
    public static class String extends KaitaiStruct {
        public static String fromFile(String fileName) throws IOException {
            return new String(new ByteBufferKaitaiStream(fileName));
        }

        public String(KaitaiStream _io) {
            this(_io, null, null);
        }

        public String(KaitaiStream _io, OpenapiMessage.Interpretation _parent) {
            this(_io, _parent, null);
        }

        public String(KaitaiStream _io, OpenapiMessage.Interpretation _parent, OpenapiMessage _root) {
            super(_io);
            this._parent = _parent;
            this._root = _root;
            _read();
        }
        private void _read() {
            this.count = this._io.readU2le();
            this.data = new String(this._io.readBytes(count()), Charset.forName("UTF-8"));
        }
        private int count;
        private String data;
        private OpenapiMessage _root;
        private OpenapiMessage.Interpretation _parent;
        public int count() { return count; }
        public String data() { return data; }
        public OpenapiMessage _root() { return _root; }
        public OpenapiMessage.Interpretation _parent() { return _parent; }
    }
    public static class TimeFamily extends KaitaiStruct {
        public static TimeFamily fromFile(String fileName) throws IOException {
            return new TimeFamily(new ByteBufferKaitaiStream(fileName));
        }

        public TimeFamily(KaitaiStream _io) {
            this(_io, null, null);
        }

        public TimeFamily(KaitaiStream _io, OpenapiMessage.Time _parent) {
            this(_io, _parent, null);
        }

        public TimeFamily(KaitaiStream _io, OpenapiMessage.Time _parent, OpenapiMessage _root) {
            super(_io);
            this._parent = _parent;
            this._root = _root;
            _read();
        }
        private void _read() {
            this.k = this._io.readU1();
            this.l = this._io.readU1();
            this.m = this._io.readU1();
            this.n = this._io.readU1();
        }
        private int k;
        private int l;
        private int m;
        private int n;
        private OpenapiMessage _root;
        private OpenapiMessage.Time _parent;
        public int k() { return k; }
        public int l() { return l; }
        public int m() { return m; }
        public int n() { return n; }
        public OpenapiMessage _root() { return _root; }
        public OpenapiMessage.Time _parent() { return _parent; }
    }
    public static class ValidityFlags extends KaitaiStruct {
        public static ValidityFlags fromFile(String fileName) throws IOException {
            return new ValidityFlags(new ByteBufferKaitaiStream(fileName));
        }

        public ValidityFlags(KaitaiStream _io) {
            this(_io, null, null);
        }

        public ValidityFlags(KaitaiStream _io, OpenapiMessage.DataQualityBlock _parent) {
            this(_io, _parent, null);
        }

        public ValidityFlags(KaitaiStream _io, OpenapiMessage.DataQualityBlock _parent, OpenapiMessage _root) {
            super(_io);
            this._parent = _parent;
            this._root = _root;
            _read();
        }
        private void _read() {
            this.f = this._io.readU2le();
        }
        private Boolean overload;
        public Boolean overload() {
            if (this.overload != null)
                return this.overload;
            boolean _tmp = (boolean) ((f() & 2) != 0);
            this.overload = _tmp;
            return this.overload;
        }
        private Boolean invalid;
        public Boolean invalid() {
            if (this.invalid != null)
                return this.invalid;
            boolean _tmp = (boolean) ((f() & 8) != 0);
            this.invalid = _tmp;
            return this.invalid;
        }
        private Boolean overrun;
        public Boolean overrun() {
            if (this.overrun != null)
                return this.overrun;
            boolean _tmp = (boolean) ((f() & 16) != 0);
            this.overrun = _tmp;
            return this.overrun;
        }
        private int f;
        private OpenapiMessage _root;
        private OpenapiMessage.DataQualityBlock _parent;
        public int f() { return f; }
        public OpenapiMessage _root() { return _root; }
        public OpenapiMessage.DataQualityBlock _parent() { return _parent; }
    }
    public static class SignalData extends KaitaiStruct {
        public static SignalData fromFile(String fileName) throws IOException {
            return new SignalData(new ByteBufferKaitaiStream(fileName));
        }

        public SignalData(KaitaiStream _io) {
            this(_io, null, null);
        }

        public SignalData(KaitaiStream _io, OpenapiMessage _parent) {
            this(_io, _parent, null);
        }

        public SignalData(KaitaiStream _io, OpenapiMessage _parent, OpenapiMessage _root) {
            super(_io);
            this._parent = _parent;
            this._root = _root;
            _read();
        }
        private void _read() {
            this.numberOfSignals = this._io.readU2le();
            this.reserved = this._io.readU2le();
            signals = new ArrayList<SignalBlock>(((Number) (numberOfSignals())).intValue());
            for (int i = 0; i < numberOfSignals(); i++) {
                this.signals.add(new SignalBlock(this._io, this, _root));
            }
        }
        private int numberOfSignals;
        private int reserved;
        private ArrayList<SignalBlock> signals;
        private OpenapiMessage _root;
        private OpenapiMessage _parent;
        public int numberOfSignals() { return numberOfSignals; }
        public int reserved() { return reserved; }
        public ArrayList<SignalBlock> signals() { return signals; }
        public OpenapiMessage _root() { return _root; }
        public OpenapiMessage _parent() { return _parent; }
    }
    public static class Header extends KaitaiStruct {
        public static Header fromFile(String fileName) throws IOException {
            return new Header(new ByteBufferKaitaiStream(fileName));
        }

        public enum EMessageType {
            E_SIGNAL_DATA(1),
            E_DATA_QUALITY(2),
            E_INTERPRETATION(8),
            E_AUX_SEQUENCE_DATA(11);

            private final long id;
            EMessageType(long id) { this.id = id; }
            public long id() { return id; }
            private static final Map<Long, EMessageType> byId = new HashMap<Long, EMessageType>(4);
            static {
                for (EMessageType e : EMessageType.values())
                    byId.put(e.id(), e);
            }
            public static EMessageType byId(long id) { return byId.get(id); }
        }

        public Header(KaitaiStream _io) {
            this(_io, null, null);
        }

        public Header(KaitaiStream _io, OpenapiMessage _parent) {
            this(_io, _parent, null);
        }

        public Header(KaitaiStream _io, OpenapiMessage _parent, OpenapiMessage _root) {
            super(_io);
            this._parent = _parent;
            this._root = _root;
            _read();
        }
        private void _read() {
            this.magic = this._io.readBytes(2);
            if (!(Arrays.equals(magic(), new byte[] { 66, 75 }))) {
                throw new KaitaiStream.ValidationNotEqualError(new byte[] { 66, 75 }, magic(), _io(), "/types/header/seq/0");
            }
            this.headerLength = this._io.readU2le();
            this.messageType = EMessageType.byId(this._io.readU2le());
            this.reserved1 = this._io.readU2le();
            this.reserved2 = this._io.readU4le();
            this.time = new Time(this._io, this, _root);
            this.messageLength = this._io.readU4le();
        }
        private byte[] magic;
        private int headerLength;
        private EMessageType messageType;
        private int reserved1;
        private long reserved2;
        private Time time;
        private long messageLength;
        private OpenapiMessage _root;
        private OpenapiMessage _parent;
        public byte[] magic() { return magic; }
        public int headerLength() { return headerLength; }
        public EMessageType messageType() { return messageType; }
        public int reserved1() { return reserved1; }
        public long reserved2() { return reserved2; }
        public Time time() { return time; }
        public long messageLength() { return messageLength; }
        public OpenapiMessage _root() { return _root; }
        public OpenapiMessage _parent() { return _parent; }
    }
    public static class SignalBlock extends KaitaiStruct {
        public static SignalBlock fromFile(String fileName) throws IOException {
            return new SignalBlock(new ByteBufferKaitaiStream(fileName));
        }

        public SignalBlock(KaitaiStream _io) {
            this(_io, null, null);
        }

        public SignalBlock(KaitaiStream _io, OpenapiMessage.SignalData _parent) {
            this(_io, _parent, null);
        }

        public SignalBlock(KaitaiStream _io, OpenapiMessage.SignalData _parent, OpenapiMessage _root) {
            super(_io);
            this._parent = _parent;
            this._root = _root;
            _read();
        }
        private void _read() {
            this.signalId = this._io.readS2le();
            this.numberOfValues = this._io.readS2le();
            values = new ArrayList<Value>(((Number) (numberOfValues())).intValue());
            for (int i = 0; i < numberOfValues(); i++) {
                this.values.add(new Value(this._io, this, _root));
            }
        }
        private short signalId;
        private short numberOfValues;
        private ArrayList<Value> values;
        private OpenapiMessage _root;
        private OpenapiMessage.SignalData _parent;
        public short signalId() { return signalId; }
        public short numberOfValues() { return numberOfValues; }
        public ArrayList<Value> values() { return values; }
        public OpenapiMessage _root() { return _root; }
        public OpenapiMessage.SignalData _parent() { return _parent; }
    }
    public static class Time extends KaitaiStruct {
        public static Time fromFile(String fileName) throws IOException {
            return new Time(new ByteBufferKaitaiStream(fileName));
        }

        public Time(KaitaiStream _io) {
            this(_io, null, null);
        }

        public Time(KaitaiStream _io, KaitaiStruct _parent) {
            this(_io, _parent, null);
        }

        public Time(KaitaiStream _io, KaitaiStruct _parent, OpenapiMessage _root) {
            super(_io);
            this._parent = _parent;
            this._root = _root;
            _read();
        }
        private void _read() {
            this.timeFamily = new TimeFamily(this._io, this, _root);
            this.timeCount = this._io.readU8le();
        }
        private TimeFamily timeFamily;
        private long timeCount;
        private OpenapiMessage _root;
        private KaitaiStruct _parent;
        public TimeFamily timeFamily() { return timeFamily; }
        public long timeCount() { return timeCount; }
        public OpenapiMessage _root() { return _root; }
        public KaitaiStruct _parent() { return _parent; }
    }
    public static class Value extends KaitaiStruct {
        public static Value fromFile(String fileName) throws IOException {
            return new Value(new ByteBufferKaitaiStream(fileName));
        }

        public Value(KaitaiStream _io) {
            this(_io, null, null);
        }

        public Value(KaitaiStream _io, OpenapiMessage.SignalBlock _parent) {
            this(_io, _parent, null);
        }

        public Value(KaitaiStream _io, OpenapiMessage.SignalBlock _parent, OpenapiMessage _root) {
            super(_io);
            this._parent = _parent;
            this._root = _root;
            _read();
        }
        private void _read() {
            this.value1 = this._io.readU1();
            this.value2 = this._io.readU1();
            this.value3 = this._io.readS1();
        }
        private Integer calcValue;
        public Integer calcValue() {
            if (this.calcValue != null)
                return this.calcValue;
            int _tmp = (int) (((value1() + (value2() << 8)) + (value3() << 16)));
            this.calcValue = _tmp;
            return this.calcValue;
        }
        private int value1;
        private int value2;
        private byte value3;
        private OpenapiMessage _root;
        private OpenapiMessage.SignalBlock _parent;
        public int value1() { return value1; }
        public int value2() { return value2; }
        public byte value3() { return value3; }
        public OpenapiMessage _root() { return _root; }
        public OpenapiMessage.SignalBlock _parent() { return _parent; }
    }
    private Header header;
    private Object message;
    private OpenapiMessage _root;
    private KaitaiStruct _parent;
    private byte[] _raw_message;
    public Header header() { return header; }
    public Object message() { return message; }
    public OpenapiMessage _root() { return _root; }
    public KaitaiStruct _parent() { return _parent; }
    public byte[] _raw_message() { return _raw_message; }
}
