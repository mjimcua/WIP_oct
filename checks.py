"""checks.py — the check recorder shared by every test main (SFF v2).

A test calls `RECORDER.check` / `RECORDER.check_raises` and moves on; the recorder
counts passes per block, keeps the text of every failure and prints the final panel.
A failing check never stops the run: one execution reports every broken check.
"""

# (no imports: pure Python)


class CheckRecorder:
    """Collect the result of every check so the runner can print one verdict.

    PURPOSE: keep the tests themselves free of bookkeeping. A test calls `check` or
    `check_raises` and moves on; the recorder knows which block is running, counts the
    passes and keeps the text of every failure for the final panel.
    """

    def __init__(self) -> None:
        self.current_block_name = "(no block)"
        self.passed_by_block: dict = {}
        self.failed_by_block: dict = {}
        self.failure_messages: list = []

    def start_block(self, block_name: str) -> None:
        """Open a block: everything checked from now on is counted under this name."""
        self.current_block_name = block_name
        self.passed_by_block.setdefault(block_name, 0)
        self.failed_by_block.setdefault(block_name, 0)
        print(f"\n{block_name}")

    def check(self, condition: bool, message: str) -> None:
        """Record one check and print it.

        INPUT:   condition — the assertion · message — what is being asserted, as a
                 sentence a reader can act on.
        OUTPUT:  none; the recorder is updated.
        RULES:   a failure is never fatal: the run continues so ONE execution reports
                 every broken check, not just the first.
        EDGE CASES: none.
        CONSOLE: one line, ok or FAIL plus the message.
        STEPS:
          [1] Print the outcome.
          [2] Count it under the current block, keeping the text if it failed.
        """
        # [1] the reader sees each check as it happens
        print(("  ok   " if condition else "  FAIL ") + message)

        # [2] bookkeeping for the final panel
        if condition:
            self.passed_by_block[self.current_block_name] += 1
        else:
            self.failed_by_block[self.current_block_name] += 1
            self.failure_messages.append(f"[{self.current_block_name}] {message}")

    def check_raises(self, callable_under_test, expected_fragment: str, message: str) -> None:
        """Record a check that passes only when the call raises ValueError NAMING the culprit.

        INPUT:   callable_under_test — zero-argument callable · expected_fragment — text
                 that must appear in the exception message · message — what is checked.
        OUTPUT:  none; recorded through `check`.
        RULES:   no exception, or an exception whose message does not name the offender,
                 is a failure. A mute ValueError is not a diagnosis.
        EDGE CASES: the fragment match is case-sensitive and substring-based; an
                 exception of another type propagates (it is a bug in the test, not a
                 failed check).
        CONSOLE: one line, through `check`, quoting the message that was raised.
        STEPS:
          [1] Call and capture the message, if any.
          [2] Require both the ValueError and the fragment in it.
        """
        # [1] run it and remember what came out
        raised_message = None
        try:
            callable_under_test()
        except ValueError as raised_error:
            raised_message = str(raised_error)

        # [2] the type AND the diagnosis matter
        has_named_diagnosis = raised_message is not None and expected_fragment in raised_message
        self.check(has_named_diagnosis, f"{message} (says: {raised_message})")

    @property
    def total_passed(self) -> int:
        return sum(self.passed_by_block.values())

    @property
    def total_failed(self) -> int:
        return sum(self.failed_by_block.values())



    def print_panel(self, title: str) -> int:
        """Print the per-block panel, the failures and the verdict; return the exit code.

        STEPS:
          [1] One line per block with passed/total.
          [2] The totals.
          [3] The failures, verbatim, and the verdict.
        """
        print("\n" + "─" * 74)
        print("PANEL BY BLOCK")

        # [1] every block with its score
        for block_name in self.passed_by_block:
            passed_count = self.passed_by_block[block_name]
            failed_count = self.failed_by_block[block_name]
            block_mark = "ok  " if failed_count == 0 else "FAIL"
            print(f"  {block_mark} {block_name:<52} {passed_count:>3}/{passed_count + failed_count}")

        # [2] the totals
        print("─" * 74)
        print(f"  TOTAL {self.total_passed}/{self.total_passed + self.total_failed} checks passed")

        # [3] what broke, in full, and the verdict
        if self.failure_messages:
            print(f"\nFAILED CHECKS ({len(self.failure_messages)}):")
            for failure_message in self.failure_messages:
                print(f"  - {failure_message}")
            print(f"\n{title} FAIL")
            return 1
        print(f"\n{title} PASS")
        return 0
