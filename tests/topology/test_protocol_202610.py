import copy
import unittest
from pcb.topology_stage.protocol_202610 import evaluate_case, logical_networks


def doc():
    return {'components': {}, 'pins': {}, 'nets': {}}


def component(d, key, kind, value=None, number='1'):
    d['components'][key] = {'Name': key, 'type': kind, 'value': value, 'bbox': [0, 0, 10, 10]}
    d['pins'][key] = {'pin_'+number: {'pinname': 'X', 'point': {'x': 5, 'y': 10}}}


def net(d, key, *refs):
    d['nets'][key] = {'hyperGraph': '('+','.join(refs)+')', 'edges': {}}


class OctoberProtocol(unittest.TestCase):
    def test_power_instance_names_and_internal_pin_numbers_do_not_define_rail(self):
        g, p = doc(), doc()
        for d, names in ((g, ('∅1', '∅2')), (p, ('PWR1', 'PWR2'))):
            for key in names:
                component(d, key, 'v', '3.3V')
            component(d, 'U1', 'box')
            component(d, 'R1', 'r')
            net(d, 'a', names[0]+'.1', 'U1.1')
            net(d, 'b', names[1]+'.1', 'R1.1')
        result = evaluate_case(p, g)
        self.assertEqual(result['metrics']['NetHypergraph']['f1'], 1)
        self.assertEqual(result['counts']['pred_logical_nets'], 1)
        self.assertEqual(result['metrics']['Component']['f1'], 1)

    def test_ordinary_shared_owner_does_not_merge_networks(self):
        d = doc()
        component(d, 'R1', 'r')
        component(d, 'U1', 'box')
        net(d, 'a', 'R1.1', 'U1.1')
        net(d, 'b', 'R1.2', 'U1.2')
        self.assertEqual(len(logical_networks(d)), 2)

    def test_missing_values_never_make_one_global_supply(self):
        d = doc()
        for key in ('P1', 'P2'):
            component(d, key, 'v')
            net(d, key, key+'.1')
        self.assertEqual(len(logical_networks(d)), 2)

    def test_voltage_text_is_exact(self):
        d = doc()
        for key, value in (('P1', '3.3V'), ('P2', '3.3v'), ('P3', ' 3.3V')):
            component(d, key, 'v', value)
            net(d, key, key+'.1')
        self.assertEqual(len(logical_networks(d)), 3)

    def test_non_ic_pins_are_not_scored_and_wrong_ic_type_does_not_hide_pins(self):
        g = doc()
        component(g, 'R1', 'r')
        component(g, 'U1', 'box')
        p = copy.deepcopy(g)
        p['pins']['R1'] = {}
        self.assertEqual(evaluate_case(p, g)['metrics']['Pin']['f1'], 1)
        p['components']['U1']['type'] = 'r'
        p['pins']['U1'] = {}
        self.assertEqual(evaluate_case(p, g)['metrics']['Pin']['gt'], 1)
        self.assertEqual(evaluate_case(p, g)['metrics']['Pin']['f1'], 0)

    def test_transitive_rail_merge_and_all_branch_edges(self):
        d = doc()
        for key, value in (('A', 'A'), ('B', 'B')):
            component(d, key, 'v', value)
        net(d, '1', 'A.1')
        net(d, '2', 'A.2', 'B.1')
        net(d, '3', 'B.2')
        line = [{'x': 0, 'y': 0}, {'x': 10, 'y': 0}]
        for row in d['nets'].values():
            row['edges']['e'] = line
        rows = logical_networks(d)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]['edges']), 3)
        self.assertEqual(evaluate_case(d, d)['metrics']['NetLine']['tp'], 3)

    def test_longest_component_prefix_and_one_to_one_symbols(self):
        d = doc()
        component(d, 'P.1', 'gnd', 'GND')
        net(d, 'n', 'P.1.9')
        p = copy.deepcopy(d)
        component(p, 'P.2', 'gnd', 'GND')
        self.assertEqual(evaluate_case(p, d)['metrics']['Component']['tp'], 1)
        self.assertEqual(logical_networks(d)[0]['vertices'], {('rail', 'GND')})


if __name__ == '__main__':
    unittest.main()
