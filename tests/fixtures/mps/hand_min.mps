* 手算验证：x=y=2，最小目标 2x+3y+5=15。
NAME sample
ROWS
 N cost
 G demand
 L cap
 E balance
COLUMNS
 x cost 2 demand 1
 x cap 1 balance 1
 y cost 3 demand 1
 y balance -1
RHS
 r demand 4 cap 3
 r balance 0 cost -5
ENDATA
