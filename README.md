# 6.A01
Robotic arm project for 6.A01 Fall 2016.

### Commands
The robot responds to the following voice commands. Commands are listed in order of priority.

Command | Function
--- | ---
dexter | Wake up the robot. No other commands will work until the robot is woken.
(dexter) calibrate | Set the current z position as the z-depth parameter.
(dexter) get position | Get the current (x, y, z) position.
(dexter) home | Go to the home position.
(dexter) info | Provide info on configuration parameters.
(dexter) ready | Asks the robot if all modules have initialized.
(dexter) sleep | Put the robot to sleep. No commands will work until the robot is woken again.
(dexter) stop | Stop the current drawing.
(dexter) draw the weather | Draw today's weather using data from Open Weather Map.
(dexter) draw the weather in `:value` `:units` | Draw the forecast using data from Open Weather Map. `:value` can be any float. `:units` can be one of second(s), minute(s), hour(s), day(s), or week(s).
(dexter) draw `*q` | Queries Pixabay for a desired SVG. `*q` can be any text.
(dexter) draw index `:i` query `*q` | Similar to draw, but allows an offset index. `:i` can be any integer. `*q` can be any text.
(dexter) trace `*q` | Queries Pixabay for a desired image. Use Potrace to convert to SVG. `*q` can be any text.
(dexter) trace index `:i` query `*q` | Similar to trace, but allows an offset index. `:i` can be any integer. `*q` can be any text.
(dexter) move `:direction` `:float` | Move the arm a certain amount in a particular direction. `:direction` can be one of up, down, left, right, forward, or backward. `:float` can be any float.
(dexter) move (to) `:x`, `:y`, `:z` | Move to a coordinate in space. `:x`, `:y`, `:z` can be any float.
(dexter) load point `:name` | Load a previously saved position. `:name` can be any text.
(dexter) save point as `:name` | Save the arm's current position under an alias. `:name` can be any text.
(dexter) set `:parameter` `:float` | Set a configuration parameter. `:parameter` can be one of lift, speed, offset, or depth. `:float` can be any float.
(dexter) `*input` | All other inputs will query Cleverbot for a response. `:input` can be any text.
