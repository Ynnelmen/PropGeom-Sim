// This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

(function (root, factory) {
  if (typeof define === 'function' && define.amd) {
    define(['kaitai-struct/KaitaiStream'], factory);
  } else if (typeof module === 'object' && module.exports) {
    module.exports = factory(require('kaitai-struct/KaitaiStream'));
  } else {
    root.OpenapiMessage = factory(root.KaitaiStream);
  }
}(this, function (KaitaiStream) {
var OpenapiMessage = (function() {
  function OpenapiMessage(_io, _parent, _root) {
    this._io = _io;
    this._parent = _parent;
    this._root = _root || this;

    this._read();
  }
  OpenapiMessage.prototype._read = function() {
    this.header = new Header(this._io, this, this._root);
    switch (this.header.messageType) {
    case OpenapiMessage.Header.EMessageType.E_SIGNAL_DATA:
      this._raw_message = this._io.readBytes(this.header.messageLength);
      var _io__raw_message = new KaitaiStream(this._raw_message);
      this.message = new SignalData(_io__raw_message, this, this._root);
      break;
    case OpenapiMessage.Header.EMessageType.E_DATA_QUALITY:
      this._raw_message = this._io.readBytes(this.header.messageLength);
      var _io__raw_message = new KaitaiStream(this._raw_message);
      this.message = new DataQuality(_io__raw_message, this, this._root);
      break;
    case OpenapiMessage.Header.EMessageType.E_INTERPRETATION:
      this._raw_message = this._io.readBytes(this.header.messageLength);
      var _io__raw_message = new KaitaiStream(this._raw_message);
      this.message = new Interpretations(_io__raw_message, this, this._root);
      break;
    default:
      this.message = this._io.readBytes(this.header.messageLength);
      break;
    }
  }

  var Interpretations = OpenapiMessage.Interpretations = (function() {
    function Interpretations(_io, _parent, _root) {
      this._io = _io;
      this._parent = _parent;
      this._root = _root || this;

      this._read();
    }
    Interpretations.prototype._read = function() {
      this.interpretations = [];
      var i = 0;
      while (!this._io.isEof()) {
        this.interpretations.push(new Interpretation(this._io, this, this._root));
        i++;
      }
    }

    return Interpretations;
  })();

  var DataQuality = OpenapiMessage.DataQuality = (function() {
    function DataQuality(_io, _parent, _root) {
      this._io = _io;
      this._parent = _parent;
      this._root = _root || this;

      this._read();
    }
    DataQuality.prototype._read = function() {
      this.numberOfSignals = this._io.readU2le();
      this.qualities = new Array(this.numberOfSignals);
      for (var i = 0; i < this.numberOfSignals; i++) {
        this.qualities[i] = new DataQualityBlock(this._io, this, this._root);
      }
    }

    return DataQuality;
  })();

  var DataQualityBlock = OpenapiMessage.DataQualityBlock = (function() {
    function DataQualityBlock(_io, _parent, _root) {
      this._io = _io;
      this._parent = _parent;
      this._root = _root || this;

      this._read();
    }
    DataQualityBlock.prototype._read = function() {
      this.signalId = this._io.readU2le();
      this.validityFlags = new ValidityFlags(this._io, this, this._root);
      this.reserved = this._io.readU2le();
    }

    return DataQualityBlock;
  })();

  var Interpretation = OpenapiMessage.Interpretation = (function() {
    Interpretation.EDescriptorType = Object.freeze({
      DATA_TYPE: 1,
      SCALE_FACTOR: 2,
      OFFSET: 3,
      PERIOD_TIME: 4,
      UNIT: 5,
      VECTOR_LENGTH: 6,
      CHANNEL_TYPE: 7,

      1: "DATA_TYPE",
      2: "SCALE_FACTOR",
      3: "OFFSET",
      4: "PERIOD_TIME",
      5: "UNIT",
      6: "VECTOR_LENGTH",
      7: "CHANNEL_TYPE",
    });

    function Interpretation(_io, _parent, _root) {
      this._io = _io;
      this._parent = _parent;
      this._root = _root || this;

      this._read();
    }
    Interpretation.prototype._read = function() {
      this.signalId = this._io.readU2le();
      this.descriptorType = this._io.readU2le();
      this.reserved = this._io.readU2le();
      this.valueLength = this._io.readU2le();
      switch (this.descriptorType) {
      case OpenapiMessage.Interpretation.EDescriptorType.DATA_TYPE:
        this.value = this._io.readU2le();
        break;
      case OpenapiMessage.Interpretation.EDescriptorType.SCALE_FACTOR:
        this.value = this._io.readF8le();
        break;
      case OpenapiMessage.Interpretation.EDescriptorType.UNIT:
        this.value = new String(this._io, this, this._root);
        break;
      case OpenapiMessage.Interpretation.EDescriptorType.VECTOR_LENGTH:
        this.value = this._io.readU2le();
        break;
      case OpenapiMessage.Interpretation.EDescriptorType.PERIOD_TIME:
        this.value = new Time(this._io, this, this._root);
        break;
      case OpenapiMessage.Interpretation.EDescriptorType.OFFSET:
        this.value = this._io.readF8le();
        break;
      case OpenapiMessage.Interpretation.EDescriptorType.CHANNEL_TYPE:
        this.value = this._io.readU2le();
        break;
      }
      this.padding = new Array(((4 - KaitaiStream.mod(this._io.pos, 4)) & 3));
      for (var i = 0; i < ((4 - KaitaiStream.mod(this._io.pos, 4)) & 3); i++) {
        this.padding[i] = this._io.readU1();
      }
    }

    return Interpretation;
  })();

  var String = OpenapiMessage.String = (function() {
    function String(_io, _parent, _root) {
      this._io = _io;
      this._parent = _parent;
      this._root = _root || this;

      this._read();
    }
    String.prototype._read = function() {
      this.count = this._io.readU2le();
      this.data = KaitaiStream.bytesToStr(this._io.readBytes(this.count), "UTF-8");
    }

    return String;
  })();

  var TimeFamily = OpenapiMessage.TimeFamily = (function() {
    function TimeFamily(_io, _parent, _root) {
      this._io = _io;
      this._parent = _parent;
      this._root = _root || this;

      this._read();
    }
    TimeFamily.prototype._read = function() {
      this.k = this._io.readU1();
      this.l = this._io.readU1();
      this.m = this._io.readU1();
      this.n = this._io.readU1();
    }

    return TimeFamily;
  })();

  var ValidityFlags = OpenapiMessage.ValidityFlags = (function() {
    function ValidityFlags(_io, _parent, _root) {
      this._io = _io;
      this._parent = _parent;
      this._root = _root || this;

      this._read();
    }
    ValidityFlags.prototype._read = function() {
      this.f = this._io.readU2le();
    }
    Object.defineProperty(ValidityFlags.prototype, 'overload', {
      get: function() {
        if (this._m_overload !== undefined)
          return this._m_overload;
        this._m_overload = (this.f & 2) != 0;
        return this._m_overload;
      }
    });
    Object.defineProperty(ValidityFlags.prototype, 'invalid', {
      get: function() {
        if (this._m_invalid !== undefined)
          return this._m_invalid;
        this._m_invalid = (this.f & 8) != 0;
        return this._m_invalid;
      }
    });
    Object.defineProperty(ValidityFlags.prototype, 'overrun', {
      get: function() {
        if (this._m_overrun !== undefined)
          return this._m_overrun;
        this._m_overrun = (this.f & 16) != 0;
        return this._m_overrun;
      }
    });

    return ValidityFlags;
  })();

  var SignalData = OpenapiMessage.SignalData = (function() {
    function SignalData(_io, _parent, _root) {
      this._io = _io;
      this._parent = _parent;
      this._root = _root || this;

      this._read();
    }
    SignalData.prototype._read = function() {
      this.numberOfSignals = this._io.readU2le();
      this.reserved = this._io.readU2le();
      this.signals = new Array(this.numberOfSignals);
      for (var i = 0; i < this.numberOfSignals; i++) {
        this.signals[i] = new SignalBlock(this._io, this, this._root);
      }
    }

    return SignalData;
  })();

  var Header = OpenapiMessage.Header = (function() {
    Header.EMessageType = Object.freeze({
      E_SIGNAL_DATA: 1,
      E_DATA_QUALITY: 2,
      E_INTERPRETATION: 8,
      E_AUX_SEQUENCE_DATA: 11,

      1: "E_SIGNAL_DATA",
      2: "E_DATA_QUALITY",
      8: "E_INTERPRETATION",
      11: "E_AUX_SEQUENCE_DATA",
    });

    function Header(_io, _parent, _root) {
      this._io = _io;
      this._parent = _parent;
      this._root = _root || this;

      this._read();
    }
    Header.prototype._read = function() {
      this.magic = this._io.readBytes(2);
      if (!((KaitaiStream.byteArrayCompare(this.magic, [66, 75]) == 0))) {
        throw new KaitaiStream.ValidationNotEqualError([66, 75], this.magic, this._io, "/types/header/seq/0");
      }
      this.headerLength = this._io.readU2le();
      this.messageType = this._io.readU2le();
      this.reserved1 = this._io.readU2le();
      this.reserved2 = this._io.readU4le();
      this.time = new Time(this._io, this, this._root);
      this.messageLength = this._io.readU4le();
    }

    return Header;
  })();

  var SignalBlock = OpenapiMessage.SignalBlock = (function() {
    function SignalBlock(_io, _parent, _root) {
      this._io = _io;
      this._parent = _parent;
      this._root = _root || this;

      this._read();
    }
    SignalBlock.prototype._read = function() {
      this.signalId = this._io.readS2le();
      this.numberOfValues = this._io.readS2le();
      this.values = new Array(this.numberOfValues);
      for (var i = 0; i < this.numberOfValues; i++) {
        this.values[i] = new Value(this._io, this, this._root);
      }
    }

    return SignalBlock;
  })();

  var Time = OpenapiMessage.Time = (function() {
    function Time(_io, _parent, _root) {
      this._io = _io;
      this._parent = _parent;
      this._root = _root || this;

      this._read();
    }
    Time.prototype._read = function() {
      this.timeFamily = new TimeFamily(this._io, this, this._root);
      this.timeCount = this._io.readU8le();
    }

    return Time;
  })();

  var Value = OpenapiMessage.Value = (function() {
    function Value(_io, _parent, _root) {
      this._io = _io;
      this._parent = _parent;
      this._root = _root || this;

      this._read();
    }
    Value.prototype._read = function() {
      this.value1 = this._io.readU1();
      this.value2 = this._io.readU1();
      this.value3 = this._io.readS1();
    }
    Object.defineProperty(Value.prototype, 'calcValue', {
      get: function() {
        if (this._m_calcValue !== undefined)
          return this._m_calcValue;
        this._m_calcValue = ((this.value1 + (this.value2 << 8)) + (this.value3 << 16));
        return this._m_calcValue;
      }
    });

    return Value;
  })();

  return OpenapiMessage;
})();
return OpenapiMessage;
}));
