# Tactics3 Stack Overflow - Isolated Position

## Random Game Position

**Game 0, Move 11** - After random moves: 117, 23, 86, 12, 81, 40, 27, 112, 18, 8, 22

Total moves played: 65 (moveCount shows board at position 65)
Current player: BLACK

Board state:
```
●·······●●●●●
·····○·●○○●··
·○·····●○○●··
·●·····●○○●··
·······●○○○●·
········●○○●·
···○··●·○●○●·
·········●○○●
········●●○○●
○········●○○●
·········●○○●
●·······●○○·○
○········●○○○
```

## Problematic Chain

**Position 22 (WHITE group at coordinate 9,1)**

- Group ID: 3
- Color: WHITE  
- Liberty count: 3
- Liberties: indices 9, 21, and one more

Board detail around position 22:
```
   ●●●●
   ●○[○]●·
   ●○○●·
   ●○○●·
```

This chain has 3 liberties (not in atari), yet still causes stack overflow.

## Why Stack Overflow Occurs

1. **searchChain()** is called on position 22
2. Current player is BLACK, defending color is WHITE (not defending)
3. **canReach4Libs()** recursively explores both capture and escape scenarios
4. **Key insight**: Even with 3 liberties (not atari), the complex board position 
   with many Black and White groups creates a very deep decision tree
5. The alternating pattern of stones forces the search to explore many branches
6. Without nodeLimit, recursion depth exceeds JavaScript call stack

## Solution

Always use `nodeLimit` parameter in `searchChains()` calls:
```javascript
searchChains(game3, 10000);  // Instead of searchChains(game3)
```

This limits recursion depth and prevents stack overflow on complex positions while still providing useful tactical analysis.

