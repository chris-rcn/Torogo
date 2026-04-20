#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, parseBoard } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');

// The failing board from the test
const boardStr = ` · ○ ● · · · ○ · ○ · · ● ●
 ● ○ · · · · · · · · · · ·
 · · · · ○ ○ · ○ · · · · ·
 · · · · · · · · · · · · ●
 ○ · ○ ● · · ○ · · · ● · ·
 · · · · · · · · ○ · · ○ ●
 · ○ ● · ● · ● ● ● ○ · ○ ·
 · · · · · · · · · · ● · ●
 · · ●(●)· · · · · · · · ·
 · · ● · · ○ · · · · · · ○
 · · ○ ○ · ● ● ● ○ · · · ·
 · ● · ○ · ○ ○ · · · ○ · ·
 ● ○ ● · · ● · · · · · ● ·`;

// Parse back from the string (which should preserve the marked position)
const g2_parsed = parseBoard(boardStr, BLACK);

// Convert to Game3
const g3 = game3FromGame2(g2_parsed);

// Get the string representations
const parsed_str = g2_parsed.toString();
const converted_str = g3.toString();

console.log('Parsed (Game2):');
console.log(parsed_str);
console.log('\nConverted (Game3):');
console.log(converted_str);

console.log('\n' + '='.repeat(50));

// Compare line by line
const pLines = parsed_str.split('\n');
const cLines = converted_str.split('\n');

for (let i = 0; i < Math.max(pLines.length, cLines.length); i++) {
  const p = pLines[i] || '';
  const c = cLines[i] || '';
  if (p !== c) {
    console.log(`Line ${i} differs:`);
    console.log(`  Parsed:    ${p}`);
    console.log(`  Converted: ${c}`);
    // Find the character position that differs
    for (let j = 0; j < Math.max(p.length, c.length); j++) {
      if (p[j] !== c[j]) {
        console.log(`    First diff at char ${j}: '${p[j]}' vs '${c[j]}'`);
        break;
      }
    }
  }
}

// Check cell states
console.log('\n' + '='.repeat(50));
console.log('Cell differences:');
let cellDiffs = 0;
for (let i = 0; i < 169; i++) {
  if (g2_parsed.cells[i] !== g3.cells[i]) {
    const y = (i / 13) | 0;
    const x = i % 13;
    console.log(`Cell ${i} (${x},${y}): Game2=${g2_parsed.cells[i]}, Game3=${g3.cells[i]}`);
    cellDiffs++;
    if (cellDiffs > 10) {
      console.log(`... and ${cellDiffs - 10} more differences`);
      break;
    }
  }
}
if (cellDiffs === 0) {
  console.log('No cell differences');
}
