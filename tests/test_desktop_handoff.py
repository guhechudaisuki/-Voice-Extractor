import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import HTTPException
from launcher import desktop_handoff as handoff, webui_server as server


class DesktopHandoffTests(unittest.TestCase):
    def setUp(self):
        mocked = patch('launcher.edge_app.edge_executable', return_value=None)
        mocked.start()
        self.addCleanup(mocked.stop)

    def launch(self, directory, payload=None, exited=False):
        process = MagicMock(pid=123)
        process.poll.return_value = 1 if exited else None

        def spawn(*args, **kwargs):
            self.assertEqual(kwargs['env']['VOICE_EXTRACT_DESKTOP_SHELL'], 'tk')
            if payload is not None:
                Path(kwargs['env'][handoff.READY_ENV]).write_text(json.dumps(payload), encoding='utf-8')
            return process

        return process, patch.object(handoff.subprocess, 'Popen', side_effect=spawn)

    def test_only_matching_visible_window_completes_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            ready = dict(pid=123, visible=True, mode='tkinter')
            process, spawn = self.launch(directory, ready)
            with spawn:
                self.assertEqual(handoff.launch_desktop(base / 'desktop.py', base), ready)
            process.terminate.assert_not_called()
            self.assertFalse(list((base / 'work/desktop_handoff').iterdir()))

    def test_missing_wrong_pid_or_hidden_window_times_out_and_cleans_child(self):
        for payload in (None, dict(pid=999, visible=True), dict(pid=123, visible=False)):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                process, spawn = self.launch(directory, payload)
                with spawn, self.assertRaisesRegex(RuntimeError, '网页仍可使用'):
                    handoff.launch_desktop(base / 'desktop.py', base, timeout=.02)
                process.terminate.assert_called_once()
                process.wait.assert_called_once()

    def test_exited_child_fails_without_waiting_for_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            process, spawn = self.launch(directory, exited=True)
            with spawn, self.assertRaises(RuntimeError):
                handoff.launch_desktop(base / 'desktop.py', base)
            process.terminate.assert_not_called()
            process.wait.assert_called_once()

    def test_unmapped_window_does_not_announce_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ready.json'
            root = MagicMock()
            root.winfo_viewable.return_value = False
            with patch.dict(handoff.os.environ, {handoff.READY_ENV: str(path)}):
                handoff.announce_ready(root)
                self.assertFalse(path.exists())
                root.winfo_viewable.return_value = True
                root.after.call_args.args[1]()
            self.assertTrue(json.loads(path.read_text())['visible'])
            root.lift.assert_called_once()


class DesktopSwitchEndpointTests(unittest.TestCase):
    def setUp(self):
        for mocked in (
            patch.object(server, '_desktop_switch_lock', threading.Lock()),
            patch.object(server.RUN, 'process', None),
        ):
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_failure_preserves_settings_and_server_and_allows_retry(self):
        with patch.object(handoff, 'launch_desktop', side_effect=RuntimeError('window timeout')), \
                patch.object(server, 'save_settings') as save, \
                patch.object(server.threading, 'Thread') as shutdown:
            with self.assertRaises(HTTPException) as error:
                server.switch_to_desktop()
            self.assertEqual(error.exception.status_code, 500)
            self.assertIn('window timeout', error.exception.detail)
            save.assert_not_called()
            shutdown.assert_not_called()
            self.assertFalse(server._desktop_switch_lock.locked())

    def test_success_waits_for_ready_then_saves_preference_and_schedules_shutdown(self):
        events = []
        ready = dict(pid=123, visible=True, mode='tkinter')
        with patch.object(handoff, 'launch_desktop', side_effect=lambda *args, **kwargs: events.append('visible') or ready), \
                patch.object(server, 'load_settings', return_value={'ui_mode': 'webui'}), \
                patch.object(server, 'save_settings', side_effect=lambda value: events.append(value['ui_mode'])), \
                patch.object(server.threading, 'Thread') as shutdown:
            shutdown.return_value.start.side_effect = lambda: events.append('shutdown')
            response = server.switch_to_desktop()
        self.assertEqual(events, ['visible', 'desktop', 'shutdown'])
        self.assertTrue(response['visible'])
        self.assertEqual(response['pid'], 123)
        self.assertTrue(server._desktop_switch_lock.locked())

    def test_running_extraction_blocks_switch(self):
        with patch.object(server.RUN, 'process', MagicMock()), \
                patch.object(handoff, 'launch_desktop') as launch:
            with self.assertRaises(HTTPException) as error:
                server.switch_to_desktop()
            self.assertEqual(error.exception.status_code, 409)
            launch.assert_not_called()
            self.assertFalse(server._desktop_switch_lock.locked())

    def test_edge_success_retains_service_and_releases_extraction_lock(self):
        ready = dict(pid=123, visible=True, mode='edge_app', retain_server=True)
        with patch.object(handoff, 'launch_desktop', return_value=ready), \
                patch.object(server, 'save_settings'), \
                patch.object(server.threading, 'Thread'):
            result = server.switch_to_desktop()
        self.assertTrue(result['retain_server'])
        self.assertFalse(server._desktop_switch_lock.locked())

    def test_switch_blocks_duplicate_click_and_new_extraction(self):
        server._desktop_switch_lock.acquire()
        for action in (server.switch_to_desktop, lambda: server.start_extract(None)):
            with self.assertRaises(HTTPException) as error:
                action()
            self.assertEqual(error.exception.status_code, 409)


if __name__ == '__main__':
    unittest.main()
