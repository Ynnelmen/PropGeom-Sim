# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

use strict;
use warnings;
use IO::KaitaiStruct 0.009_000;
use Encode;

########################################################################
package OpenapiMessage;

our @ISA = 'IO::KaitaiStruct::Struct';

sub from_file {
    my ($class, $filename) = @_;
    my $fd;

    open($fd, '<', $filename) or return undef;
    binmode($fd);
    return new($class, IO::KaitaiStruct::Stream->new($fd));
}

sub new {
    my ($class, $_io, $_parent, $_root) = @_;
    my $self = IO::KaitaiStruct::Struct->new($_io);

    bless $self, $class;
    $self->{_parent} = $_parent;
    $self->{_root} = $_root || $self;;

    $self->_read();

    return $self;
}

sub _read {
    my ($self) = @_;

    $self->{header} = OpenapiMessage::Header->new($self->{_io}, $self, $self->{_root});
    my $_on = $self->header()->message_type();
    if ($_on == $OpenapiMessage::Header::E_MESSAGE_TYPE_E_SIGNAL_DATA) {
        $self->{_raw_message} = $self->{_io}->read_bytes($self->header()->message_length());
        my $io__raw_message = IO::KaitaiStruct::Stream->new($self->{_raw_message});
        $self->{message} = OpenapiMessage::SignalData->new($io__raw_message, $self, $self->{_root});
    }
    elsif ($_on == $OpenapiMessage::Header::E_MESSAGE_TYPE_E_DATA_QUALITY) {
        $self->{_raw_message} = $self->{_io}->read_bytes($self->header()->message_length());
        my $io__raw_message = IO::KaitaiStruct::Stream->new($self->{_raw_message});
        $self->{message} = OpenapiMessage::DataQuality->new($io__raw_message, $self, $self->{_root});
    }
    elsif ($_on == $OpenapiMessage::Header::E_MESSAGE_TYPE_E_INTERPRETATION) {
        $self->{_raw_message} = $self->{_io}->read_bytes($self->header()->message_length());
        my $io__raw_message = IO::KaitaiStruct::Stream->new($self->{_raw_message});
        $self->{message} = OpenapiMessage::Interpretations->new($io__raw_message, $self, $self->{_root});
    }
    else {
        $self->{message} = $self->{_io}->read_bytes($self->header()->message_length());
    }
}

sub header {
    my ($self) = @_;
    return $self->{header};
}

sub message {
    my ($self) = @_;
    return $self->{message};
}

sub _raw_message {
    my ($self) = @_;
    return $self->{_raw_message};
}

########################################################################
package OpenapiMessage::Interpretations;

our @ISA = 'IO::KaitaiStruct::Struct';

sub from_file {
    my ($class, $filename) = @_;
    my $fd;

    open($fd, '<', $filename) or return undef;
    binmode($fd);
    return new($class, IO::KaitaiStruct::Stream->new($fd));
}

sub new {
    my ($class, $_io, $_parent, $_root) = @_;
    my $self = IO::KaitaiStruct::Struct->new($_io);

    bless $self, $class;
    $self->{_parent} = $_parent;
    $self->{_root} = $_root || $self;;

    $self->_read();

    return $self;
}

sub _read {
    my ($self) = @_;

    $self->{interpretations} = ();
    while (!$self->{_io}->is_eof()) {
        push @{$self->{interpretations}}, OpenapiMessage::Interpretation->new($self->{_io}, $self, $self->{_root});
    }
}

sub interpretations {
    my ($self) = @_;
    return $self->{interpretations};
}

########################################################################
package OpenapiMessage::DataQuality;

our @ISA = 'IO::KaitaiStruct::Struct';

sub from_file {
    my ($class, $filename) = @_;
    my $fd;

    open($fd, '<', $filename) or return undef;
    binmode($fd);
    return new($class, IO::KaitaiStruct::Stream->new($fd));
}

sub new {
    my ($class, $_io, $_parent, $_root) = @_;
    my $self = IO::KaitaiStruct::Struct->new($_io);

    bless $self, $class;
    $self->{_parent} = $_parent;
    $self->{_root} = $_root || $self;;

    $self->_read();

    return $self;
}

sub _read {
    my ($self) = @_;

    $self->{number_of_signals} = $self->{_io}->read_u2le();
    $self->{qualities} = ();
    my $n_qualities = $self->number_of_signals();
    for (my $i = 0; $i < $n_qualities; $i++) {
        $self->{qualities}[$i] = OpenapiMessage::DataQualityBlock->new($self->{_io}, $self, $self->{_root});
    }
}

sub number_of_signals {
    my ($self) = @_;
    return $self->{number_of_signals};
}

sub qualities {
    my ($self) = @_;
    return $self->{qualities};
}

########################################################################
package OpenapiMessage::DataQualityBlock;

our @ISA = 'IO::KaitaiStruct::Struct';

sub from_file {
    my ($class, $filename) = @_;
    my $fd;

    open($fd, '<', $filename) or return undef;
    binmode($fd);
    return new($class, IO::KaitaiStruct::Stream->new($fd));
}

sub new {
    my ($class, $_io, $_parent, $_root) = @_;
    my $self = IO::KaitaiStruct::Struct->new($_io);

    bless $self, $class;
    $self->{_parent} = $_parent;
    $self->{_root} = $_root || $self;;

    $self->_read();

    return $self;
}

sub _read {
    my ($self) = @_;

    $self->{signal_id} = $self->{_io}->read_u2le();
    $self->{validity_flags} = OpenapiMessage::ValidityFlags->new($self->{_io}, $self, $self->{_root});
    $self->{reserved} = $self->{_io}->read_u2le();
}

sub signal_id {
    my ($self) = @_;
    return $self->{signal_id};
}

sub validity_flags {
    my ($self) = @_;
    return $self->{validity_flags};
}

sub reserved {
    my ($self) = @_;
    return $self->{reserved};
}

########################################################################
package OpenapiMessage::Interpretation;

our @ISA = 'IO::KaitaiStruct::Struct';

sub from_file {
    my ($class, $filename) = @_;
    my $fd;

    open($fd, '<', $filename) or return undef;
    binmode($fd);
    return new($class, IO::KaitaiStruct::Stream->new($fd));
}

our $E_DESCRIPTOR_TYPE_DATA_TYPE = 1;
our $E_DESCRIPTOR_TYPE_SCALE_FACTOR = 2;
our $E_DESCRIPTOR_TYPE_OFFSET = 3;
our $E_DESCRIPTOR_TYPE_PERIOD_TIME = 4;
our $E_DESCRIPTOR_TYPE_UNIT = 5;
our $E_DESCRIPTOR_TYPE_VECTOR_LENGTH = 6;
our $E_DESCRIPTOR_TYPE_CHANNEL_TYPE = 7;

sub new {
    my ($class, $_io, $_parent, $_root) = @_;
    my $self = IO::KaitaiStruct::Struct->new($_io);

    bless $self, $class;
    $self->{_parent} = $_parent;
    $self->{_root} = $_root || $self;;

    $self->_read();

    return $self;
}

sub _read {
    my ($self) = @_;

    $self->{signal_id} = $self->{_io}->read_u2le();
    $self->{descriptor_type} = $self->{_io}->read_u2le();
    $self->{reserved} = $self->{_io}->read_u2le();
    $self->{value_length} = $self->{_io}->read_u2le();
    my $_on = $self->descriptor_type();
    if ($_on == $OpenapiMessage::Interpretation::E_DESCRIPTOR_TYPE_DATA_TYPE) {
        $self->{value} = $self->{_io}->read_u2le();
    }
    elsif ($_on == $OpenapiMessage::Interpretation::E_DESCRIPTOR_TYPE_SCALE_FACTOR) {
        $self->{value} = $self->{_io}->read_f8le();
    }
    elsif ($_on == $OpenapiMessage::Interpretation::E_DESCRIPTOR_TYPE_UNIT) {
        $self->{value} = OpenapiMessage::String->new($self->{_io}, $self, $self->{_root});
    }
    elsif ($_on == $OpenapiMessage::Interpretation::E_DESCRIPTOR_TYPE_VECTOR_LENGTH) {
        $self->{value} = $self->{_io}->read_u2le();
    }
    elsif ($_on == $OpenapiMessage::Interpretation::E_DESCRIPTOR_TYPE_PERIOD_TIME) {
        $self->{value} = OpenapiMessage::Time->new($self->{_io}, $self, $self->{_root});
    }
    elsif ($_on == $OpenapiMessage::Interpretation::E_DESCRIPTOR_TYPE_OFFSET) {
        $self->{value} = $self->{_io}->read_f8le();
    }
    elsif ($_on == $OpenapiMessage::Interpretation::E_DESCRIPTOR_TYPE_CHANNEL_TYPE) {
        $self->{value} = $self->{_io}->read_u2le();
    }
    $self->{padding} = ();
    my $n_padding = ((4 - ($self->_io()->pos() % 4)) & 3);
    for (my $i = 0; $i < $n_padding; $i++) {
        $self->{padding}[$i] = $self->{_io}->read_u1();
    }
}

sub signal_id {
    my ($self) = @_;
    return $self->{signal_id};
}

sub descriptor_type {
    my ($self) = @_;
    return $self->{descriptor_type};
}

sub reserved {
    my ($self) = @_;
    return $self->{reserved};
}

sub value_length {
    my ($self) = @_;
    return $self->{value_length};
}

sub value {
    my ($self) = @_;
    return $self->{value};
}

sub padding {
    my ($self) = @_;
    return $self->{padding};
}

########################################################################
package OpenapiMessage::String;

our @ISA = 'IO::KaitaiStruct::Struct';

sub from_file {
    my ($class, $filename) = @_;
    my $fd;

    open($fd, '<', $filename) or return undef;
    binmode($fd);
    return new($class, IO::KaitaiStruct::Stream->new($fd));
}

sub new {
    my ($class, $_io, $_parent, $_root) = @_;
    my $self = IO::KaitaiStruct::Struct->new($_io);

    bless $self, $class;
    $self->{_parent} = $_parent;
    $self->{_root} = $_root || $self;;

    $self->_read();

    return $self;
}

sub _read {
    my ($self) = @_;

    $self->{count} = $self->{_io}->read_u2le();
    $self->{data} = Encode::decode("UTF-8", $self->{_io}->read_bytes($self->count()));
}

sub count {
    my ($self) = @_;
    return $self->{count};
}

sub data {
    my ($self) = @_;
    return $self->{data};
}

########################################################################
package OpenapiMessage::TimeFamily;

our @ISA = 'IO::KaitaiStruct::Struct';

sub from_file {
    my ($class, $filename) = @_;
    my $fd;

    open($fd, '<', $filename) or return undef;
    binmode($fd);
    return new($class, IO::KaitaiStruct::Stream->new($fd));
}

sub new {
    my ($class, $_io, $_parent, $_root) = @_;
    my $self = IO::KaitaiStruct::Struct->new($_io);

    bless $self, $class;
    $self->{_parent} = $_parent;
    $self->{_root} = $_root || $self;;

    $self->_read();

    return $self;
}

sub _read {
    my ($self) = @_;

    $self->{k} = $self->{_io}->read_u1();
    $self->{l} = $self->{_io}->read_u1();
    $self->{m} = $self->{_io}->read_u1();
    $self->{n} = $self->{_io}->read_u1();
}

sub k {
    my ($self) = @_;
    return $self->{k};
}

sub l {
    my ($self) = @_;
    return $self->{l};
}

sub m {
    my ($self) = @_;
    return $self->{m};
}

sub n {
    my ($self) = @_;
    return $self->{n};
}

########################################################################
package OpenapiMessage::ValidityFlags;

our @ISA = 'IO::KaitaiStruct::Struct';

sub from_file {
    my ($class, $filename) = @_;
    my $fd;

    open($fd, '<', $filename) or return undef;
    binmode($fd);
    return new($class, IO::KaitaiStruct::Stream->new($fd));
}

sub new {
    my ($class, $_io, $_parent, $_root) = @_;
    my $self = IO::KaitaiStruct::Struct->new($_io);

    bless $self, $class;
    $self->{_parent} = $_parent;
    $self->{_root} = $_root || $self;;

    $self->_read();

    return $self;
}

sub _read {
    my ($self) = @_;

    $self->{f} = $self->{_io}->read_u2le();
}

sub overload {
    my ($self) = @_;
    return $self->{overload} if ($self->{overload});
    $self->{overload} = ($self->f() & 2) != 0;
    return $self->{overload};
}

sub invalid {
    my ($self) = @_;
    return $self->{invalid} if ($self->{invalid});
    $self->{invalid} = ($self->f() & 8) != 0;
    return $self->{invalid};
}

sub overrun {
    my ($self) = @_;
    return $self->{overrun} if ($self->{overrun});
    $self->{overrun} = ($self->f() & 16) != 0;
    return $self->{overrun};
}

sub f {
    my ($self) = @_;
    return $self->{f};
}

########################################################################
package OpenapiMessage::SignalData;

our @ISA = 'IO::KaitaiStruct::Struct';

sub from_file {
    my ($class, $filename) = @_;
    my $fd;

    open($fd, '<', $filename) or return undef;
    binmode($fd);
    return new($class, IO::KaitaiStruct::Stream->new($fd));
}

sub new {
    my ($class, $_io, $_parent, $_root) = @_;
    my $self = IO::KaitaiStruct::Struct->new($_io);

    bless $self, $class;
    $self->{_parent} = $_parent;
    $self->{_root} = $_root || $self;;

    $self->_read();

    return $self;
}

sub _read {
    my ($self) = @_;

    $self->{number_of_signals} = $self->{_io}->read_u2le();
    $self->{reserved} = $self->{_io}->read_u2le();
    $self->{signals} = ();
    my $n_signals = $self->number_of_signals();
    for (my $i = 0; $i < $n_signals; $i++) {
        $self->{signals}[$i] = OpenapiMessage::SignalBlock->new($self->{_io}, $self, $self->{_root});
    }
}

sub number_of_signals {
    my ($self) = @_;
    return $self->{number_of_signals};
}

sub reserved {
    my ($self) = @_;
    return $self->{reserved};
}

sub signals {
    my ($self) = @_;
    return $self->{signals};
}

########################################################################
package OpenapiMessage::Header;

our @ISA = 'IO::KaitaiStruct::Struct';

sub from_file {
    my ($class, $filename) = @_;
    my $fd;

    open($fd, '<', $filename) or return undef;
    binmode($fd);
    return new($class, IO::KaitaiStruct::Stream->new($fd));
}

our $E_MESSAGE_TYPE_E_SIGNAL_DATA = 1;
our $E_MESSAGE_TYPE_E_DATA_QUALITY = 2;
our $E_MESSAGE_TYPE_E_INTERPRETATION = 8;
our $E_MESSAGE_TYPE_E_AUX_SEQUENCE_DATA = 11;

sub new {
    my ($class, $_io, $_parent, $_root) = @_;
    my $self = IO::KaitaiStruct::Struct->new($_io);

    bless $self, $class;
    $self->{_parent} = $_parent;
    $self->{_root} = $_root || $self;;

    $self->_read();

    return $self;
}

sub _read {
    my ($self) = @_;

    $self->{magic} = $self->{_io}->read_bytes(2);
    $self->{header_length} = $self->{_io}->read_u2le();
    $self->{message_type} = $self->{_io}->read_u2le();
    $self->{reserved1} = $self->{_io}->read_u2le();
    $self->{reserved2} = $self->{_io}->read_u4le();
    $self->{time} = OpenapiMessage::Time->new($self->{_io}, $self, $self->{_root});
    $self->{message_length} = $self->{_io}->read_u4le();
}

sub magic {
    my ($self) = @_;
    return $self->{magic};
}

sub header_length {
    my ($self) = @_;
    return $self->{header_length};
}

sub message_type {
    my ($self) = @_;
    return $self->{message_type};
}

sub reserved1 {
    my ($self) = @_;
    return $self->{reserved1};
}

sub reserved2 {
    my ($self) = @_;
    return $self->{reserved2};
}

sub time {
    my ($self) = @_;
    return $self->{time};
}

sub message_length {
    my ($self) = @_;
    return $self->{message_length};
}

########################################################################
package OpenapiMessage::SignalBlock;

our @ISA = 'IO::KaitaiStruct::Struct';

sub from_file {
    my ($class, $filename) = @_;
    my $fd;

    open($fd, '<', $filename) or return undef;
    binmode($fd);
    return new($class, IO::KaitaiStruct::Stream->new($fd));
}

sub new {
    my ($class, $_io, $_parent, $_root) = @_;
    my $self = IO::KaitaiStruct::Struct->new($_io);

    bless $self, $class;
    $self->{_parent} = $_parent;
    $self->{_root} = $_root || $self;;

    $self->_read();

    return $self;
}

sub _read {
    my ($self) = @_;

    $self->{signal_id} = $self->{_io}->read_s2le();
    $self->{number_of_values} = $self->{_io}->read_s2le();
    $self->{values} = ();
    my $n_values = $self->number_of_values();
    for (my $i = 0; $i < $n_values; $i++) {
        $self->{values}[$i] = OpenapiMessage::Value->new($self->{_io}, $self, $self->{_root});
    }
}

sub signal_id {
    my ($self) = @_;
    return $self->{signal_id};
}

sub number_of_values {
    my ($self) = @_;
    return $self->{number_of_values};
}

sub values {
    my ($self) = @_;
    return $self->{values};
}

########################################################################
package OpenapiMessage::Time;

our @ISA = 'IO::KaitaiStruct::Struct';

sub from_file {
    my ($class, $filename) = @_;
    my $fd;

    open($fd, '<', $filename) or return undef;
    binmode($fd);
    return new($class, IO::KaitaiStruct::Stream->new($fd));
}

sub new {
    my ($class, $_io, $_parent, $_root) = @_;
    my $self = IO::KaitaiStruct::Struct->new($_io);

    bless $self, $class;
    $self->{_parent} = $_parent;
    $self->{_root} = $_root || $self;;

    $self->_read();

    return $self;
}

sub _read {
    my ($self) = @_;

    $self->{time_family} = OpenapiMessage::TimeFamily->new($self->{_io}, $self, $self->{_root});
    $self->{time_count} = $self->{_io}->read_u8le();
}

sub time_family {
    my ($self) = @_;
    return $self->{time_family};
}

sub time_count {
    my ($self) = @_;
    return $self->{time_count};
}

########################################################################
package OpenapiMessage::Value;

our @ISA = 'IO::KaitaiStruct::Struct';

sub from_file {
    my ($class, $filename) = @_;
    my $fd;

    open($fd, '<', $filename) or return undef;
    binmode($fd);
    return new($class, IO::KaitaiStruct::Stream->new($fd));
}

sub new {
    my ($class, $_io, $_parent, $_root) = @_;
    my $self = IO::KaitaiStruct::Struct->new($_io);

    bless $self, $class;
    $self->{_parent} = $_parent;
    $self->{_root} = $_root || $self;;

    $self->_read();

    return $self;
}

sub _read {
    my ($self) = @_;

    $self->{value1} = $self->{_io}->read_u1();
    $self->{value2} = $self->{_io}->read_u1();
    $self->{value3} = $self->{_io}->read_s1();
}

sub calc_value {
    my ($self) = @_;
    return $self->{calc_value} if ($self->{calc_value});
    $self->{calc_value} = (($self->value1() + ($self->value2() << 8)) + ($self->value3() << 16));
    return $self->{calc_value};
}

sub value1 {
    my ($self) = @_;
    return $self->{value1};
}

sub value2 {
    my ($self) = @_;
    return $self->{value2};
}

sub value3 {
    my ($self) = @_;
    return $self->{value3};
}

1;
