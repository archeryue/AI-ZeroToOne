"""Move dataclass for Chinese Chess."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Move:
    from_row: int
    from_col: int
    to_row: int
    to_col: int
    captured: int = 0  # piece code that was captured (0 = no capture)

    def to_ucci(self) -> str:
        """Convert to UCCI-style 4-char string, e.g. 'b2e2'."""
        col_chars = "abcdefghi"
        return (
            f"{col_chars[self.from_col]}{self.from_row}"
            f"{col_chars[self.to_col]}{self.to_row}"
        )

    @classmethod
    def from_ucci(cls, s: str) -> "Move":
        """Parse a UCCI-style string like 'b2e2'."""
        col_chars = "abcdefghi"
        from_col = col_chars.index(s[0])
        from_row = int(s[1])
        to_col = col_chars.index(s[2])
        to_row = int(s[3])
        return cls(from_row, from_col, to_row, to_col)

    def __repr__(self) -> str:
        return f"Move({self.from_row},{self.from_col}->{self.to_row},{self.to_col})"
