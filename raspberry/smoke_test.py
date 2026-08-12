"""Hardware-free smoke test for Raspberry Pi deployment."""

import time

from app import SafeLiftApp, create_app


def main() -> None:
    worker = SafeLiftApp(mock=True, camera_index=0, no_camera=True)
    import threading
    thread = threading.Thread(target=worker.loop, daemon=True)
    thread.start()
    time.sleep(1.0)
    client = create_app(worker).test_client()
    state = client.get('/api/state').get_json()
    health = client.get('/api/health').get_json()
    worker.close()
    assert state['sensor']['valid']
    assert 0 <= state['score']['score'] <= 100
    assert state['risk']['level'] in {'normal', 'warning', 'danger'}
    assert health['mock'] is True
    print('SMOKE_TEST_PASS')
    print({'score': state['score']['score'], 'risk': state['risk']['level']})


if __name__ == '__main__':
    main()
