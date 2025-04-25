# Just some humanoid dynamics that corresponds with the failure states.
# angles = radians; time = seconds; distances = meters

# Conditions that can cause failure:

# Format:   Link Name       Conditional     Threshold

# Pelvis Height is basically on the floor. 
# 1.        "Root" Z-Coord      <               0.5

# Rotation and Pitch are calculated from the Sim quaternion
# Robot is rotated too far in any direction:
# 2.        abs("Root" Roll)    >               0.7
# 3.        abs("Root" Pitch)   >               0.7

#           Left_Foot_Grounded  and Right_Foot_Grounded are true when 
#           sim detects feet collision with ground plane.
# -         Airborn      !=   (Left_Foot_Grounded | Right_Foot_Grounded)
# -         Start Timer when airborn becomes true.
# -         Air_Time     = Time in air

# Robot is airborn for too long and pelvis is too low:
# 4.        (Air_Time  >  0.3) & ("Root" Z-Coord < 0.7)


# Any joint with limits are out of bounds:
# 5.         Knee_Joint_State   <       -3.14   - 0.05
# 6.         Knee_Joint_State   >       0      + 0.05
# 7.         Elbow_Joint_State  <       0      - 0.05
# 8.         Elbow_Joint_State  >       3.14   + 0.05

