<?php
// This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

namespace {
    class OpenapiMessage extends \Kaitai\Struct\Struct {
        public function __construct(\Kaitai\Struct\Stream $_io, \Kaitai\Struct\Struct $_parent = null, \OpenapiMessage $_root = null) {
            parent::__construct($_io, $_parent, $_root);
            $this->_read();
        }

        private function _read() {
            $this->_m_header = new \OpenapiMessage\Header($this->_io, $this, $this->_root);
            switch ($this->header()->messageType()) {
                case \OpenapiMessage\Header\EMessageType::E_SIGNAL_DATA:
                    $this->_m__raw_message = $this->_io->readBytes($this->header()->messageLength());
                    $_io__raw_message = new \Kaitai\Struct\Stream($this->_m__raw_message);
                    $this->_m_message = new \OpenapiMessage\SignalData($_io__raw_message, $this, $this->_root);
                    break;
                case \OpenapiMessage\Header\EMessageType::E_DATA_QUALITY:
                    $this->_m__raw_message = $this->_io->readBytes($this->header()->messageLength());
                    $_io__raw_message = new \Kaitai\Struct\Stream($this->_m__raw_message);
                    $this->_m_message = new \OpenapiMessage\DataQuality($_io__raw_message, $this, $this->_root);
                    break;
                case \OpenapiMessage\Header\EMessageType::E_INTERPRETATION:
                    $this->_m__raw_message = $this->_io->readBytes($this->header()->messageLength());
                    $_io__raw_message = new \Kaitai\Struct\Stream($this->_m__raw_message);
                    $this->_m_message = new \OpenapiMessage\Interpretations($_io__raw_message, $this, $this->_root);
                    break;
                default:
                    $this->_m_message = $this->_io->readBytes($this->header()->messageLength());
                    break;
            }
        }
        protected $_m_header;
        protected $_m_message;
        protected $_m__raw_message;
        public function header() { return $this->_m_header; }
        public function message() { return $this->_m_message; }
        public function _raw_message() { return $this->_m__raw_message; }
    }
}

namespace OpenapiMessage {
    class Interpretations extends \Kaitai\Struct\Struct {
        public function __construct(\Kaitai\Struct\Stream $_io, \OpenapiMessage $_parent = null, \OpenapiMessage $_root = null) {
            parent::__construct($_io, $_parent, $_root);
            $this->_read();
        }

        private function _read() {
            $this->_m_interpretations = [];
            $i = 0;
            while (!$this->_io->isEof()) {
                $this->_m_interpretations[] = new \OpenapiMessage\Interpretation($this->_io, $this, $this->_root);
                $i++;
            }
        }
        protected $_m_interpretations;
        public function interpretations() { return $this->_m_interpretations; }
    }
}

namespace OpenapiMessage {
    class DataQuality extends \Kaitai\Struct\Struct {
        public function __construct(\Kaitai\Struct\Stream $_io, \OpenapiMessage $_parent = null, \OpenapiMessage $_root = null) {
            parent::__construct($_io, $_parent, $_root);
            $this->_read();
        }

        private function _read() {
            $this->_m_numberOfSignals = $this->_io->readU2le();
            $this->_m_qualities = [];
            $n = $this->numberOfSignals();
            for ($i = 0; $i < $n; $i++) {
                $this->_m_qualities[] = new \OpenapiMessage\DataQualityBlock($this->_io, $this, $this->_root);
            }
        }
        protected $_m_numberOfSignals;
        protected $_m_qualities;
        public function numberOfSignals() { return $this->_m_numberOfSignals; }
        public function qualities() { return $this->_m_qualities; }
    }
}

namespace OpenapiMessage {
    class DataQualityBlock extends \Kaitai\Struct\Struct {
        public function __construct(\Kaitai\Struct\Stream $_io, \OpenapiMessage\DataQuality $_parent = null, \OpenapiMessage $_root = null) {
            parent::__construct($_io, $_parent, $_root);
            $this->_read();
        }

        private function _read() {
            $this->_m_signalId = $this->_io->readU2le();
            $this->_m_validityFlags = new \OpenapiMessage\ValidityFlags($this->_io, $this, $this->_root);
            $this->_m_reserved = $this->_io->readU2le();
        }
        protected $_m_signalId;
        protected $_m_validityFlags;
        protected $_m_reserved;
        public function signalId() { return $this->_m_signalId; }
        public function validityFlags() { return $this->_m_validityFlags; }
        public function reserved() { return $this->_m_reserved; }
    }
}

namespace OpenapiMessage {
    class Interpretation extends \Kaitai\Struct\Struct {
        public function __construct(\Kaitai\Struct\Stream $_io, \OpenapiMessage\Interpretations $_parent = null, \OpenapiMessage $_root = null) {
            parent::__construct($_io, $_parent, $_root);
            $this->_read();
        }

        private function _read() {
            $this->_m_signalId = $this->_io->readU2le();
            $this->_m_descriptorType = $this->_io->readU2le();
            $this->_m_reserved = $this->_io->readU2le();
            $this->_m_valueLength = $this->_io->readU2le();
            switch ($this->descriptorType()) {
                case \OpenapiMessage\Interpretation\EDescriptorType::DATA_TYPE:
                    $this->_m_value = $this->_io->readU2le();
                    break;
                case \OpenapiMessage\Interpretation\EDescriptorType::SCALE_FACTOR:
                    $this->_m_value = $this->_io->readF8le();
                    break;
                case \OpenapiMessage\Interpretation\EDescriptorType::UNIT:
                    $this->_m_value = new \OpenapiMessage\String($this->_io, $this, $this->_root);
                    break;
                case \OpenapiMessage\Interpretation\EDescriptorType::VECTOR_LENGTH:
                    $this->_m_value = $this->_io->readU2le();
                    break;
                case \OpenapiMessage\Interpretation\EDescriptorType::PERIOD_TIME:
                    $this->_m_value = new \OpenapiMessage\Time($this->_io, $this, $this->_root);
                    break;
                case \OpenapiMessage\Interpretation\EDescriptorType::OFFSET:
                    $this->_m_value = $this->_io->readF8le();
                    break;
                case \OpenapiMessage\Interpretation\EDescriptorType::CHANNEL_TYPE:
                    $this->_m_value = $this->_io->readU2le();
                    break;
            }
            $this->_m_padding = [];
            $n = ((4 - \Kaitai\Struct\Stream::mod($this->_io()->pos(), 4)) & 3);
            for ($i = 0; $i < $n; $i++) {
                $this->_m_padding[] = $this->_io->readU1();
            }
        }
        protected $_m_signalId;
        protected $_m_descriptorType;
        protected $_m_reserved;
        protected $_m_valueLength;
        protected $_m_value;
        protected $_m_padding;
        public function signalId() { return $this->_m_signalId; }
        public function descriptorType() { return $this->_m_descriptorType; }
        public function reserved() { return $this->_m_reserved; }
        public function valueLength() { return $this->_m_valueLength; }
        public function value() { return $this->_m_value; }
        public function padding() { return $this->_m_padding; }
    }
}

namespace OpenapiMessage\Interpretation {
    class EDescriptorType {
        const DATA_TYPE = 1;
        const SCALE_FACTOR = 2;
        const OFFSET = 3;
        const PERIOD_TIME = 4;
        const UNIT = 5;
        const VECTOR_LENGTH = 6;
        const CHANNEL_TYPE = 7;
    }
}

namespace OpenapiMessage {
    class String extends \Kaitai\Struct\Struct {
        public function __construct(\Kaitai\Struct\Stream $_io, \OpenapiMessage\Interpretation $_parent = null, \OpenapiMessage $_root = null) {
            parent::__construct($_io, $_parent, $_root);
            $this->_read();
        }

        private function _read() {
            $this->_m_count = $this->_io->readU2le();
            $this->_m_data = \Kaitai\Struct\Stream::bytesToStr($this->_io->readBytes($this->count()), "UTF-8");
        }
        protected $_m_count;
        protected $_m_data;
        public function count() { return $this->_m_count; }
        public function data() { return $this->_m_data; }
    }
}

namespace OpenapiMessage {
    class TimeFamily extends \Kaitai\Struct\Struct {
        public function __construct(\Kaitai\Struct\Stream $_io, \OpenapiMessage\Time $_parent = null, \OpenapiMessage $_root = null) {
            parent::__construct($_io, $_parent, $_root);
            $this->_read();
        }

        private function _read() {
            $this->_m_k = $this->_io->readU1();
            $this->_m_l = $this->_io->readU1();
            $this->_m_m = $this->_io->readU1();
            $this->_m_n = $this->_io->readU1();
        }
        protected $_m_k;
        protected $_m_l;
        protected $_m_m;
        protected $_m_n;
        public function k() { return $this->_m_k; }
        public function l() { return $this->_m_l; }
        public function m() { return $this->_m_m; }
        public function n() { return $this->_m_n; }
    }
}

namespace OpenapiMessage {
    class ValidityFlags extends \Kaitai\Struct\Struct {
        public function __construct(\Kaitai\Struct\Stream $_io, \OpenapiMessage\DataQualityBlock $_parent = null, \OpenapiMessage $_root = null) {
            parent::__construct($_io, $_parent, $_root);
            $this->_read();
        }

        private function _read() {
            $this->_m_f = $this->_io->readU2le();
        }
        protected $_m_overload;
        public function overload() {
            if ($this->_m_overload !== null)
                return $this->_m_overload;
            $this->_m_overload = ($this->f() & 2) != 0;
            return $this->_m_overload;
        }
        protected $_m_invalid;
        public function invalid() {
            if ($this->_m_invalid !== null)
                return $this->_m_invalid;
            $this->_m_invalid = ($this->f() & 8) != 0;
            return $this->_m_invalid;
        }
        protected $_m_overrun;
        public function overrun() {
            if ($this->_m_overrun !== null)
                return $this->_m_overrun;
            $this->_m_overrun = ($this->f() & 16) != 0;
            return $this->_m_overrun;
        }
        protected $_m_f;
        public function f() { return $this->_m_f; }
    }
}

namespace OpenapiMessage {
    class SignalData extends \Kaitai\Struct\Struct {
        public function __construct(\Kaitai\Struct\Stream $_io, \OpenapiMessage $_parent = null, \OpenapiMessage $_root = null) {
            parent::__construct($_io, $_parent, $_root);
            $this->_read();
        }

        private function _read() {
            $this->_m_numberOfSignals = $this->_io->readU2le();
            $this->_m_reserved = $this->_io->readU2le();
            $this->_m_signals = [];
            $n = $this->numberOfSignals();
            for ($i = 0; $i < $n; $i++) {
                $this->_m_signals[] = new \OpenapiMessage\SignalBlock($this->_io, $this, $this->_root);
            }
        }
        protected $_m_numberOfSignals;
        protected $_m_reserved;
        protected $_m_signals;
        public function numberOfSignals() { return $this->_m_numberOfSignals; }
        public function reserved() { return $this->_m_reserved; }
        public function signals() { return $this->_m_signals; }
    }
}

namespace OpenapiMessage {
    class Header extends \Kaitai\Struct\Struct {
        public function __construct(\Kaitai\Struct\Stream $_io, \OpenapiMessage $_parent = null, \OpenapiMessage $_root = null) {
            parent::__construct($_io, $_parent, $_root);
            $this->_read();
        }

        private function _read() {
            $this->_m_magic = $this->_io->readBytes(2);
            if (!($this->magic() == "\x42\x4B")) {
                throw new \Kaitai\Struct\Error\ValidationNotEqualError("\x42\x4B", $this->magic(), $this->_io(), "/types/header/seq/0");
            }
            $this->_m_headerLength = $this->_io->readU2le();
            $this->_m_messageType = $this->_io->readU2le();
            $this->_m_reserved1 = $this->_io->readU2le();
            $this->_m_reserved2 = $this->_io->readU4le();
            $this->_m_time = new \OpenapiMessage\Time($this->_io, $this, $this->_root);
            $this->_m_messageLength = $this->_io->readU4le();
        }
        protected $_m_magic;
        protected $_m_headerLength;
        protected $_m_messageType;
        protected $_m_reserved1;
        protected $_m_reserved2;
        protected $_m_time;
        protected $_m_messageLength;
        public function magic() { return $this->_m_magic; }
        public function headerLength() { return $this->_m_headerLength; }
        public function messageType() { return $this->_m_messageType; }
        public function reserved1() { return $this->_m_reserved1; }
        public function reserved2() { return $this->_m_reserved2; }
        public function time() { return $this->_m_time; }
        public function messageLength() { return $this->_m_messageLength; }
    }
}

namespace OpenapiMessage\Header {
    class EMessageType {
        const E_SIGNAL_DATA = 1;
        const E_DATA_QUALITY = 2;
        const E_INTERPRETATION = 8;
        const E_AUX_SEQUENCE_DATA = 11;
    }
}

namespace OpenapiMessage {
    class SignalBlock extends \Kaitai\Struct\Struct {
        public function __construct(\Kaitai\Struct\Stream $_io, \OpenapiMessage\SignalData $_parent = null, \OpenapiMessage $_root = null) {
            parent::__construct($_io, $_parent, $_root);
            $this->_read();
        }

        private function _read() {
            $this->_m_signalId = $this->_io->readS2le();
            $this->_m_numberOfValues = $this->_io->readS2le();
            $this->_m_values = [];
            $n = $this->numberOfValues();
            for ($i = 0; $i < $n; $i++) {
                $this->_m_values[] = new \OpenapiMessage\Value($this->_io, $this, $this->_root);
            }
        }
        protected $_m_signalId;
        protected $_m_numberOfValues;
        protected $_m_values;
        public function signalId() { return $this->_m_signalId; }
        public function numberOfValues() { return $this->_m_numberOfValues; }
        public function values() { return $this->_m_values; }
    }
}

namespace OpenapiMessage {
    class Time extends \Kaitai\Struct\Struct {
        public function __construct(\Kaitai\Struct\Stream $_io, \Kaitai\Struct\Struct $_parent = null, \OpenapiMessage $_root = null) {
            parent::__construct($_io, $_parent, $_root);
            $this->_read();
        }

        private function _read() {
            $this->_m_timeFamily = new \OpenapiMessage\TimeFamily($this->_io, $this, $this->_root);
            $this->_m_timeCount = $this->_io->readU8le();
        }
        protected $_m_timeFamily;
        protected $_m_timeCount;
        public function timeFamily() { return $this->_m_timeFamily; }
        public function timeCount() { return $this->_m_timeCount; }
    }
}

namespace OpenapiMessage {
    class Value extends \Kaitai\Struct\Struct {
        public function __construct(\Kaitai\Struct\Stream $_io, \OpenapiMessage\SignalBlock $_parent = null, \OpenapiMessage $_root = null) {
            parent::__construct($_io, $_parent, $_root);
            $this->_read();
        }

        private function _read() {
            $this->_m_value1 = $this->_io->readU1();
            $this->_m_value2 = $this->_io->readU1();
            $this->_m_value3 = $this->_io->readS1();
        }
        protected $_m_calcValue;
        public function calcValue() {
            if ($this->_m_calcValue !== null)
                return $this->_m_calcValue;
            $this->_m_calcValue = (($this->value1() + ($this->value2() << 8)) + ($this->value3() << 16));
            return $this->_m_calcValue;
        }
        protected $_m_value1;
        protected $_m_value2;
        protected $_m_value3;
        public function value1() { return $this->_m_value1; }
        public function value2() { return $this->_m_value2; }
        public function value3() { return $this->_m_value3; }
    }
}
