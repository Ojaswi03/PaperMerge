from basil_core.cart_notification_text import event


def test_complete_message_is_spaced_and_hides_long_filesystem_paths():
    title,message=event('complete','/home/user/research/.papermerge/local-tests/cart-six',completed=6,total=6,nodes=100,rounds=100)
    assert title=='CART run complete' and 'Completed: 6 / 6\nFailed: 0' in message
    assert '\n\nClients: 100\nRounds per configuration: 100\n\n' in message
    assert 'Run: cart-six' in message and '/home/' not in message and 'Traceback' not in message


def test_ready_message_has_one_clear_commands_link_and_commit():
    title,message=event('ready','cifar-next',commit='0123456789abcdef',commands_url='https://example.test/commands')
    assert title=='CART ready for HPC'
    assert 'Commit: 0123456789ab' in message and '\n\nCommands:\nhttps://example.test/commands\n\n' in message


def test_failure_and_interruption_alerts_remain_short_and_actionable():
    for kind in ('failed','interrupted','worker_failed','partial','report_failed'):
        _,message=event(kind,'/home/user/project/.papermerge/local-tests/cifar6')
        assert len(message)<300 and '\n\n' in message and '/home/' not in message
        assert 'Run: cifar6' in message
    assert 'Other configurations are continuing.' in event('worker_failed','cifar6')[1]
