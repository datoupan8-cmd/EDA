"""Official raw-target type taxonomy from the 210-case annotation audit."""

OFFICIAL_COMPONENT_TYPES = frozenset({
    'amp','amp_3pin','amp_5pin','ant','battary','bjt','bjt_npn','bjt_pnp',
    'block','box','buzzer','c','circle_header','crystal','crystal_2pin',
    'crystal_3pin','crystal_4pin','d','dc','esd','fuse','gnd','jumper','l','led',
    'm3螺丝','mosfet','mosfet_npn','mosfet_pnp','motor','net_bidirection',
    'net_input','net_output','net_short','opt','other','pin','r','seg','switch',
    'testpoint','transfomer','v',
})

OLD_TO_OFFICIAL_TYPE = {
    'resistor':'r','capacitor':'c','inductor':'l','ic':'box','vcc':'v',
    'diode':'d','connector':'pin','battery':'battary','amplifier':'amp',
    'vbus_in':'net_input','vbus_out':'net_output','vbus_bi':'net_bidirection',
}

def to_official_type(value: str) -> str:
    mapped = OLD_TO_OFFICIAL_TYPE.get(str(value), str(value))
    return mapped if mapped in OFFICIAL_COMPONENT_TYPES else 'other'
