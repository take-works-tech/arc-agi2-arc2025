# DSL Command Quick Reference (Implemented Commands Only)

**Total Commands**: 89 commands + 3 control structures + 1 global function (as of 2025-12-10) ⭐v4.1

> Source: Built directly from the `execute_command` implementation in `src/core_systems/executor/parsing/interpreter.py`. Names, arguments, and return values mirror the runtime.

## Category Summary

| Category | Commands | Summary |
|---------|----------|---------|
| Object Transformations | 22 | Move, reshape, and recolour objects |
| Generation / Extraction / Split | 10 | Create, merge, extract, and split objects |
| Relations & Distances | 5 | Compute relationships between objects |
| Information Retrieval | 26 | Query sizes, positions, colours, and statistics |
| Array & Set Operations | 11 | Manipulate lists and structured collections |
| Predicate Functions | 4 | Return boolean conditions |
| Arithmetic Operations | 5 | Integer arithmetic |
| Comparison Operations | 4 | Integer comparisons |
| Logical Operations | 2 | Boolean logic |
| Global Functions | 1 | `RENDER_GRID` |
| Control Structures | 3 | `FOR` / `WHILE` / `IF` |

Each category is detailed below. Argument order matches the runtime implementation and return types describe the value produced by the interpreter.

---

## Object Transformations (22)

| Command | Arguments (type) | Returns (type) | Description |
|---------|------------------|----------------|-------------|
| `MOVE` | obj (str); dx (int); dy (int) | str (object_id) | Translate an object by offsets |
| `TELEPORT` | obj (str); x (int); y (int) | str (object_id) | Move to absolute coordinates |
| `SLIDE` | obj (str); direction (str); obstacles (List[str]) | str (object_id) | Slide until hitting obstacles |
| `ROTATE` | obj (str); angle (int); cx (int, optional); cy (int, optional) | str (object_id) | Rotate in 90ﾂｰ increments (optional pivot) |
| `FLIP` | obj (str); axis (str) | str (object_id) | Flip horizontally or vertically |
| `SCALE` | obj (str); factor (int) | str (object_id) | Upscale coordinates by an integer factor |
| `SCALE_DOWN` | obj (str); divisor (int) | str (object_id) | Downscale uniformly (1 / divisor) |
| `EXPAND` | obj (str); pixels (int) | str (object_id) | Expand the object by pixel thickness |
| `FLOW` | obj (str); direction (str); obstacles (List[str]) | str (object_id) | Flow downward or sideways like liquid |
| `DRAW` | obj (str); x (int); y (int) | str (object_id) | Draw a trace at specified coordinates |
| `LAY` | obj (str); direction (str); obstacles (List[str]) | str (object_id) | Drop with gravity-like behaviour |
| `CROP` | obj (str); x (int); y (int); w (int); h (int) | str (object_id) | Crop by rectangle |
| `SET_COLOR` | obj (str); color (int) | str (object_id) | Recolour with a single value |
| `FILL_HOLES` | obj (str); color (int) | str (object_id) | Fill interior holes |
| `OUTLINE` | obj (str); color (int) | str (object_id) | Extract the outline and paint it |
| `HOLLOW` | obj (str) | str (object_id) | Leave the border only |
| `FIT_SHAPE` | obj1 (str); obj2 (str) | str (object_id) | Shape-fit `obj1` onto `obj2` |
| `FIT_SHAPE_COLOR` | obj1 (str); obj2 (str) | str (object_id) | Fit shape and colour simultaneously |
| `FIT_ADJACENT` | obj1 (str); obj2 (str) | str (object_id) | Optimise adjacent placement |
| `ALIGN` | obj (str); mode (str) | str (object_id) | Align object (left/right/top/bottom/center_x/center_y/center) |
| `PATHFIND` | obj (str); target_x (int); target_y (int); obstacles (List[str]) | str (object_id) | Move via pathfinding |

Directional commands (`SLIDE` / `FLOW` / `LAY`) accept `"X"`, `"Y"`, `"-X"`, `"-Y"`. `ROTATE` honours 0/90/180/270 degrees. `SCALE` and `SCALE_DOWN` require integer factors.

---

## Generation / Extraction / Split (10)

| Command | Arguments (type) | Returns (type) | Description |
|---------|------------------|----------------|-------------|
| `MERGE` | objects (List[str]) | str (object_id) | Merge objects (order preserved) into a new object |
| `CREATE_LINE` | x (int); y (int); length (int); direction (str); color (int) | str (object_id) | Generate line objects (supports 8 directions) |
| `CREATE_RECT` | x (int); y (int); w (int); h (int); color (int) | str (object_id) | Create filled rectangles |
| `EXTRACT_BBOX` | obj (str); color (int) | str (object_id) | Extract bounding box and draw in colour |
| `EXTRACT_RECTS` | obj (str) | List[str] | Extract rectangular segments |
| `EXTRACT_HOLLOW_RECTS` | obj (str) | List[str] | Extract hollow rectangles only |
| `EXTRACT_LINES` | obj (str) | List[str] | Extract linear segments |
| `SPLIT_CONNECTED` | obj (str); connectivity (int {4,8}) | List[str] | Split into connected components |
| `BBOX` | obj (str); color (int, optional) | str (object_id) | Extract bounding box (color optional) |
| `TILE` | obj (str); count_x (int); count_y (int) | List[str] | Tile object in a grid pattern |

`CREATE_LINE` accepts `"X"`, `"Y"`, `"-X"`, `"-Y"`, `"XY"`, `"-XY"`, `"X-Y"`, `"-X-Y"` for direction. `MERGE` preserves input order in the resulting ID.

---

## Relations & Distances (5)

| Command | Arguments (type) | Returns (type) | Description |
|---------|------------------|----------------|-------------|
| `INTERSECTION` | obj1 (str); obj2 (str) | str (object_id) | Extract overlapping region |
| `SUBTRACT` | obj1 (str); obj2 (str) | str (object_id) | Remove `obj2` pixels from `obj1` |
| `COUNT_HOLES` | obj (str) | int | Count the number of holes |
| `COUNT_ADJACENT` | obj1 (str); obj2 (str) | int | Count touching pixels along the boundary |
| `COUNT_OVERLAP` | obj1 (str); obj2 (str) | int | Count overlapping pixels |

---

## Information Retrieval (26)

| Command | Arguments (type) | Returns (type) | Description |
|---------|------------------|----------------|-------------|
| `GET_ALL_OBJECTS` | connectivity (int {4,8}) | List[str] | List all object IDs with given connectivity |
| `GET_BACKGROUND_COLOR` | - | int | Background colour of the input grid |
| `GET_INPUT_GRID_SIZE` | - | List[int] | `[width, height]`
| `GET_SIZE` | obj (str) | int | Total pixel count |
| `GET_WIDTH` | obj (str) | int | Width in pixels |
| `GET_HEIGHT` | obj (str) | int | Height in pixels |
| `GET_X` | obj (str) | int | Leftmost X coordinate |
| `GET_Y` | obj (str) | int | Topmost Y coordinate |
| `GET_COLOR` | obj (str) | int | Majority colour |
| `GET_COLORS` | obj (str) | List[int] | List of colours present |
| `GET_SYMMETRY_SCORE` | obj (str); axis (str {"X","Y"}) | int | Symmetry score (0窶・00) |
| `GET_LINE_TYPE` | obj (str) | str | `"X"`, `"Y"`, `"XY"`, `"-XY"`, `"none"` |
| `GET_RECTANGLE_TYPE` | obj (str) | str | `"solid"` or `"hollow"` |
| `GET_DISTANCE` | obj1 (str); obj2 (str) | int | Euclidean distance |
| `GET_X_DISTANCE` | obj1 (str); obj2 (str) | int | Distance along X |
| `GET_Y_DISTANCE` | obj1 (str); obj2 (str) | int | Distance along Y |
| `GET_ASPECT_RATIO` | obj (str) | int | Aspect ratio (integer × 100) |
| `GET_DENSITY` | obj (str) | int | Density (integer × 100) |
| `GET_CENTER_X` | obj (str) | int | Center X coordinate |
| `GET_CENTER_Y` | obj (str) | int | Center Y coordinate |
| `GET_MAX_X` | obj (str) | int | Rightmost X coordinate |
| `GET_MAX_Y` | obj (str) | int | Bottommost Y coordinate |
| `GET_CENTROID` | obj (str) | str | Centroid direction ("X", "Y", "C", etc.) |
| `GET_DIRECTION` | obj1 (str); obj2 (str) | str | Direction from obj1 to obj2 |
| `GET_NEAREST` | obj (str); candidates (List[str]) | str (object_id) | Get nearest object |

---

## Array & Set Operations (11)

| Command | Arguments (type) | Returns (type) | Description |
|---------|------------------|----------------|-------------|
| `APPEND` | array_name (str); value (Any) | List[Any] | Append value to named array (non-destructive) |
| `LEN` | array (List[Any]) | int | Length of an array |
| `REVERSE` | array (List[Any]) | List[Any] | Reverse array (returns new array) |
| `CONCAT` | array1 (List[Any]); array2 (List[Any]) | List[Any] | Concatenate two arrays |
| `FILTER` | objects (List[str]); condition_expr (AST) | List[str] | Filter with an expression |
| `SORT_BY` | array (List[str]); key_expr (AST); order (str {"asc","desc"}) | List[str] | Stable sort by key |
| `EXTEND_PATTERN` | objects (List[str]); side (str {"front","end"}); count (int) | List[str] | Repeat pattern on either side |
| `ARRANGE_GRID` | objects (List[str]); columns (int); cell_width (int); cell_height (int) | List[str] | Arrange IDs in a grid layout |
| `MATCH_PAIRS` | objects1 (List[str]); objects2 (List[str]); condition_expr (AST) | List[str] | Extract pairs matching a condition |
| `EXCLUDE` | array (List[str]); targets (List[str]) | List[str] | Exclude fully matching objects |
| `CREATE_ARRAY` | - | List[Any] | Create a new empty array |

`FILTER` and `SORT_BY` receive AST nodes; use `$obj` within expressions. `APPEND` expects a variable name (identifier) as its first argument.

---

## Predicate Functions (4)

| Command | Arguments (type) | Returns (type) | Description |
|---------|------------------|----------------|-------------|
| `IS_INSIDE` | obj (str); x (int); y (int); width (int); height (int) | bool | Check if object lies fully inside a rectangle |
| `IS_SAME_SHAPE` | obj1 (str); obj2 (str) | bool | Compare shapes |
| `IS_SAME_STRUCT` | obj1 (str); obj2 (str) | bool | Compare shape and colour structure |
| `IS_IDENTICAL` | obj1 (str); obj2 (str) | bool | Exact pixel match |

---

## Numeric, Comparison, and Logical Operations

| Group | Commands | Arguments (type) | Returns (type) | Description |
|-------|----------|------------------|----------------|-------------|
| Arithmetic (5) | `ADD`, `SUB`, `MULTIPLY`, `DIVIDE`, `MOD` | a (int); b (int) | int | Integer arithmetic only |
| Comparison (4) | `EQUAL`, `NOT_EQUAL`, `GREATER`, `LESS` | a (int); b (int) | bool | Strict integer comparisons |
| Logical (2) | `AND`, `OR` | a (bool); b (bool) | bool | Boolean logic |

---

## Global Function & Control Structures

| Command / Structure | Arguments (type) | Returns (type) | Description |
|---------------------|------------------|----------------|-------------|
| `RENDER_GRID` | objects (List[str]); bg (int); width (int); height (int); x (int, optional); y (int, optional) | None | Finalise the output grid and store it in `execution_context` |
| `FOR` | var (identifier); count (int) | None | Repeat up to 1,000 iterations |
| `WHILE` | condition (AST) | None | Loop while condition holds |
| `IF` | condition (AST); then_block (Block); else_block (Block, optional) | None | Conditional execution |

Calling `RENDER_GRID` sets `program_terminated`, skipping subsequent nodes. `FOR` throws `SilentException` if object limits are exceeded.

---

## Placeholder and Expression Rules

- `$obj` references the current object ID inside `FILTER`, `SORT_BY`, and `MATCH_PAIRS` expressions.
- `APPEND` treats the first argument as a variable name; do not pass string literals.
- `SORT_BY` is stable. Use `order="asc"` or `order="desc"`; the result is a new array.
- `EXCLUDE` performs exact matches (coordinates, size, pixels). `None` values are automatically ignored.

---

## Notes on `GET` Commands

- Always include parentheses for zero-argument commands (`GET_BACKGROUND_COLOR()`), otherwise the DSL parser throws a syntax error.
- `GET_LINE_TYPE` can return `"none"`; provide fallbacks when branching on its value.
- `GET_COLORS` returns colours in arbitrary order; post-process if ordering matters.

---

## Runtime Constraints

- Arrays containing object IDs cannot exceed `MAX_OBJECTS_IN_VARIABLE` (see `src/core_systems/executor/core.py`).
- `EXCLUDE` and `FILTER` log warnings when handling large collections; adjust log levels during profiling.
- All arithmetic commands accept integers only. Passing floats raises `TypeError`.

---

## Revision History

- **v4.1 (2025-12-10)**: Re-audited interpreter implementation and updated to 89 commands. Added `REVERSE`, `ALIGN`, `PATHFIND`, `BBOX`, `TILE`, and additional information retrieval commands (`GET_ASPECT_RATIO`, `GET_DENSITY`, `GET_CENTER_X/Y`, `GET_MAX_X/Y`, `GET_CENTROID`, `GET_DIRECTION`, `GET_NEAREST`).
- **v4.0 (2025-11-10)**: Reclassified 74 commands after auditing the interpreter implementation. Added notes on `CREATE_ARRAY` and relation commands.
- **v3.8 (2025-11-10)**: Post-script clean-up overview (legacy).

Update this table whenever new DSL commands land in `src/core_systems/executor/parsing/interpreter.py`.
