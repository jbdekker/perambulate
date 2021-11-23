# Modifying conditions

- ``move``  
  Shift the intervals of the Condition forward or backward
  
  
  ```python
  # shift forwards by 5 days
  C.move("5d")
  
  # shift backwards by 20 minutes
  C.move("-20m")  
  ```
  
- ``grow``  
  Grow one or both ends of each interval
  
  ```python
  # grow both ends by 5 days
  C.grow("5d", "both")  
  
  # grow the left side by 1 day
  C.grow("1d", "left")  
  
  # grow the right side by 2 minutes
  C.grow("2min", "right")
  ```
  
- ``grow_end``  
  Extend each interval to the right as far as possible i.e. until it touches the next interval or the end of the timeseries
  
  ```python
  C.grow_end()
  ```

- ``shrink``  
  The inverse of ``grow``
  
- ``reduce``  
  Minimize the number of intervals by merge any overlapping intervals
  
  ```python
  C.reduce()
  ```
