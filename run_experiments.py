
#!/usr/bin/env python3


from experiments import Experiment1, Experiment2, Experiment3


OPTIONS = {
	"1": {"key": "TDA", "label": "TDA — Effectiveness"},
	"2": {"key": "TDAQ", "label": "TDAQ — Scalability: High"},
	"3": {"key": "TSLD", "label": "TSLD — Effectiveness: Moderate"},

}


def prompt_choice():
	"""Prompt the user until a valid choice is entered and return the key."""
	print("Select one option by number or name:")
	for num in ("1", "2", "3"):
		print(f" {num}. {OPTIONS[num]['label']}")

	while True:
		try:
			choice = input("Your choice: ").strip().lower()
		except (EOFError, KeyboardInterrupt):
			print("\nInput cancelled. Exiting.")
			raise SystemExit(1)

		if choice in OPTIONS:
			sel = OPTIONS[choice]
			print(f"Selected: {sel['key']} — {sel['label']}")
			return sel['key']

		print("Invalid choice. Enter 1, 2, 3, or TDA/TDAQ/TSLD.")


MAPPING = {
	"TDA": Experiment1,
	"TDAQ": Experiment2,
	"TSLD": Experiment3,
}


def main():
	
	selected = prompt_choice()
	cls = MAPPING.get(selected)
	exp = cls(config={})
	exp.setup()
	out = exp.run()
	exp.report()



if __name__ == "__main__":
	main()

