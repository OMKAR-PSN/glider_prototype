"""Fix unicode characters in test file for Windows cp1252 terminal compatibility."""
with open('tests/test_actuation_uav_dataset.py', 'r', encoding='utf-8') as f:
    content = f.read()

fixes = [
    ('\u2193', '[DL]'), ('\u2713', '[OK]'), ('\u2717', '[FAIL]'),
    ('\u2714', '[OK]'), ('\u2716', '[FAIL]'), ('\u2192', '->'),
    ('\u2500', '-'), ('\u2501', '-'), ('\u2502', '|'), ('\u2014', '--'),
    ('\u2013', '-'), ('\u00b0', 'deg'), ('\u2019', "'"), ('\u201c', '"'),
    ('\u201d', '"'), ('\u2192', '->'), ('\u2190', '<-'), ('\u2022', '*'),
]
for old, new in fixes:
    content = content.replace(old, new)

# Replace any remaining non-ASCII safely
result = content.encode('ascii', 'replace').decode('ascii')

with open('tests/test_actuation_uav_dataset.py', 'w', encoding='utf-8') as f:
    f.write(result)

bad = [i+1 for i, l in enumerate(result.splitlines()) if any(ord(c) > 127 for c in l)]
print(f"Remaining non-ASCII lines: {len(bad)}")
if bad:
    print(f"Lines: {bad[:10]}")
else:
    print("All clean! Ready to run.")
