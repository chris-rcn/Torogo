# Tactics3 Stack Overflow - Isolated Position

## Position
Playing the first 30 legal moves on a 13×13 board creates an alternating pattern in the top row:

```
○●○●○●○●○●○●○
●·●·●○●○●○●○●
○●○●·········
·············
·············
·············
······●······
·············
·············
·············
·············
·············
·············
```

Move sequence: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29

## Problematic Chain

**Position 2 (WHITE stone at coordinate 0,2)**

- Group ID: 2
- Color: WHITE  
- Liberty count: 1 (in ATARI)
- Single liberty: index 158 (location not on visible board)
- Current player: BLACK (attacker)

Nearby board detail:
```
○●[○]
●·●
○●○
```

## Why Stack Overflow Occurs

1. **searchChain()** is called on the WHITE stone at position 2
2. Because current player (BLACK) ≠ defending color (WHITE) and atari is true:
   - Evaluates if BLACK can force capture (moverSucceeds)
3. **canReach4Libs()** recursively explores moves with unlimited depth:
   - Each liberty creates branches in the search tree
   - No budget limit means recursion continues until finding conclusion
4. **The alternating row pattern** creates complex ladder-like positions where:
   - Each move creates new groups with limited liberties
   - Both capture and escape scenarios require deep exploration
   - Recursion depth exceeds JavaScript call stack (~10,000 calls)

## Solution

Always use `nodeLimit` parameter in `searchChains()` calls:
```javascript
searchChains(game3, 10000);  // Instead of searchChains(game3)
```

This limits recursion depth and prevents stack overflow on complex ladder positions while still providing useful tactical analysis for most practical board positions.
