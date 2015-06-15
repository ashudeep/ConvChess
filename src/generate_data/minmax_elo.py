import pgn, sys
minelo = 4000
maxelo = 0000
for game in pgn.GameIterator(sys.argv[1]):
	if not game:	
		break
	blackelo = int(game.blackelo)
	whiteelo = int(game.whiteelo)
	if blackelo <minelo:
		minelo = (blackelo)
	if whiteelo <minelo:
		minelo = whiteelo
	if blackelo >maxelo:
		maxelo = blackelo
	if whiteelo>maxelo:
		maxelo = whiteelo

print "Min ELO %d"%minelo
print "Max ELO %d"%maxelo