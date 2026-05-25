# GUI CSS Parsing Fix

## Problem
When running the GUI, it showed: "CSS parsing failed: 3 errors found in stylesheet"

## Root Cause
The Textual CSS had invalid syntax:
- `grid-template-columns` and `grid-template-rows` are not valid Textual properties (these are CSS Grid, not Textual)
- `grid-size: 3 4` syntax was incorrect
- Complex grid spanning on a Screen-level grid caused parsing issues

## Solution

### Changed from:
```css
Screen {
    layout: grid;
    grid-template-columns: 1fr 1fr 2fr;
    grid-template-rows: auto auto 1fr 1fr;
    gap: 1;
}
#input-container { column-span: 3; height: auto; border: solid white; }
#job-info { row-span: 1; column-span: 1; border: solid white; }
...
```

### To:
```css
Screen {
    layout: vertical;
}
Header { dock: top; height: 1; }
Footer { dock: bottom; height: 1; }
#input-container { 
    height: auto;
    border: solid white;
    padding: 1;
}
#main-grid {
    layout: grid;
    grid-size: 3 3;
    height: 1fr;
    gap: 1;
}
#job-info { border: solid white; }
#agent-status { row-span: 2; border: solid cyan; }
...
```

### Layout Changes:
- Screen uses `vertical` layout (stacking components)
- Input container and main grid are siblings in vertical layout
- Main grid is a nested container with `layout: grid`
- Grid properties (row-span, etc.) now valid within grid container

### compose() Method Update:
```python
def compose(self) -> ComposeResult:
    yield Header()
    with self._input_container:
        yield self.goal_input
        yield self.submit_btn
    with Container(id="main-grid"):
        yield self.job_info
        yield self.agent_status
        yield self.file_tree
        yield self.progress
        yield self.raw_output
    yield Footer()
```

## Testing
✓ CSS parsing: SUCCESS
✓ Dashboard instantiation: SUCCESS  
✓ No parsing errors reported

## Result
The GUI now starts without CSS errors and displays correctly with:
- Input area at top (goal entry)
- Dashboard panels in grid below
- Proper spacing and borders
