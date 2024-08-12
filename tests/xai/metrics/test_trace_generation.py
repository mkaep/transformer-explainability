

import unittest

from processtransformer.xai.metrics.trace_generation import generate_real_local_env, replace_single_event
from tests.xai.metrics.common import _generate_df_from_events


class TestTraceGeneration(unittest.TestCase):

    def test_generate_real_local_env1(self):
        log = _generate_df_from_events([
            ['A', 'C', 'D', 'F', 'G', 'H'],
            ['A', 'C', 'E', 'F', 'G', 'H'],
            ['A', 'B', 'A', 'C', 'D', 'F', 'G', 'H'],
            ['A', 'B', 'A', 'B', 'A', 'C', 'D', 'F', 'G', 'H'],
        ])
        res = generate_real_local_env(['A', 'C', 'D', 'F', 'G', 'H'], log)
        assert len(res) == 2
        assert ['A', 'C', 'E', 'F', 'G', 'H'] in res
        assert ['A', 'B', 'A', 'C', 'D', 'F', 'G', 'H'] in res
        pass

    def test_generate_real_local_env2(self):
        log = _generate_df_from_events([
            ['A', 'B', 'C'],
        ])
        res = generate_real_local_env(['A', 'B', 'C'], log)
        assert len(res) == 0
        pass

    def test_generate_real_local_env3(self):
        log = _generate_df_from_events([
            ['A', 'B', 'D', 'E'],
            ['A', 'C', 'D', 'E'],
        ])
        res = generate_real_local_env(['A', 'B', 'D', 'E'], log)
        assert len(res) == 1
        assert ['A', 'C', 'D', 'E'] in res
        pass

    def test_generate_real_local_env4(self):
        log = _generate_df_from_events([
            ['A', 'B', 'C'],
            ['A', 'A', 'B', 'C'],
            ['A', 'A', 'A', 'B', 'C'],
            ['A', 'A', 'A', 'A', 'B', 'C'],
        ])
        res = generate_real_local_env(['A', 'B', 'C'], log)
        assert ['A', 'A', 'B', 'C'] in res
        pass

    def test_generate_real_local_env5(self):
        log = _generate_df_from_events([
            ['A', 'B', 'D', 'E'],
            ['A', 'C', 'D', 'E'],
        ])
        res = generate_real_local_env(['A', 'B', 'D', 'E'], log, return_base_trace=True)
        assert len(res) == 2
        assert ['A', 'B', 'D', 'E'] in res
        assert ['A', 'C', 'D', 'E'] in res

        res = generate_real_local_env(['A', 'B', 'D', 'E'], log, return_base_trace=False)
        assert len(res) == 1
        assert ['A', 'B', 'D', 'E'] not in res
        assert ['A', 'C', 'D', 'E'] in res
        pass


class TestSingleEventReplacement(unittest.TestCase):
    def test_replace_single_event1(self):
        log = _generate_df_from_events([
            ['A', 'B', 'C', 'F', 'G'],
            ['A', 'B', 'D', 'F', 'G'],
            ['A', 'B', 'E', 'F', 'G'],
        ])
        res = replace_single_event(['A', 'B', 'C', 'F', 'G'], index=2, event_log=log)
        assert len(res) == 3
        assert ['A', 'B', 'M-C', 'F', 'G'] in res
        assert ['A', 'B', 'D', 'F', 'G'] in res
        assert ['A', 'B', 'E', 'F', 'G'] in res
        pass

    def test_replace_single_event2(self):
        log = _generate_df_from_events([
            ['A', 'B', 'C', 'F', 'G'],
        ])
        res = replace_single_event(['A', 'B', 'C', 'F', 'G'], index=2, event_log=log)
        assert len(res) == 1
        assert ['A', 'B', 'M-C', 'F', 'G'] in res
        pass

    def test_replace_single_event3(self):
        log = _generate_df_from_events([
            ['A', 'X', 'Y'],
            ['B', 'X', 'Y'],
            ['C', 'X', 'Y'],
        ])
        res = replace_single_event(['A', 'X', 'Y'], index=0, event_log=log)
        assert len(res) == 3
        assert ['M-A', 'X', 'Y'] in res
        assert ['B', 'X', 'Y'] in res
        assert ['C', 'X', 'Y'] in res
        pass

    def test_replace_single_event4(self):
        log = _generate_df_from_events([
            ['A', 'B', 'U'],
            ['A', 'B', 'V'],
            ['A', 'B', 'W'],
        ])
        res = replace_single_event(['A', 'B', 'U'], index=2, event_log=log)
        assert len(res) == 3
        assert ['A', 'B', 'M-U'] in res
        assert ['A', 'B', 'V'] in res
        assert ['A', 'B', 'W'] in res
        pass

    def test_replace_single_event5(self):
        log = _generate_df_from_events([
            ['A', 'B', 'C', 'F', 'G'],
            ['A', 'B', 'C', 'F', 'G'],
            ['A', 'B', 'E', 'C', 'G'],
        ])
        res = replace_single_event(['A', 'B', 'C', 'F', 'G'], index=2, event_log=log)
        assert len(res) == 1
        assert ['A', 'B', 'M-C', 'F', 'G'] in res
        pass

    def test_replace_single_event6(self):
        log = _generate_df_from_events([
            ['A', 'B', 'C', 'F', 'G'],
            ['A', 'B', 'D', 'F', 'G'],
            ['A', 'B', 'E', 'F', 'G'],
        ])
        res = replace_single_event(['A', 'B', 'C', 'F', 'G'], index=1, event_log=log)
        assert len(res) == 1
        assert ['A', 'M-B', 'C', 'F', 'G'] in res
        pass


if __name__ == '__main__':
    unittest.main()
