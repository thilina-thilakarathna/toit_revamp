
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
	if cls is None:
		print(f"No experiment class mapped for selection: {selected}")
		raise SystemExit(2)

	# Instantiate and run the selected experiment. Replace `None` with
	# a real dataset or configuration when integrating into the pipeline.
	exp = cls(config={})
	# If the user selected TDA (Experiment1) provide cleaned data
	if selected == "TDA":

		out = exp.run()
		# Print basic metrics if present
		metrics = out.get('metrics') if isinstance(out, dict) else None
		print("Metrics:", metrics)
	elif selected == "TDAQ":
		# Experiment2: timing/scalability
		try:
			out = exp.run()
			avg_tda = out.get('avg_time_tda') if isinstance(out, dict) else None
			avg_ifum = out.get('avg_time_ifum') if isinstance(out, dict) else None
			print("Average time (TDA):")
			print(avg_tda)
			print("Average time (IFUM):")
			print(avg_ifum)
		except Exception as e:
			print("Experiment2 failed:", e)
	elif selected == "TSLD":
		# Experiment3: effectiveness sweep
		try:
			out = exp.run()
			# `out` is a DataFrame; show a small summary
			try:
				print(out.head())
			except Exception:
				print(out)
		except Exception as e:
			print("Experiment3 failed:", e)
	else:
		# fallback: run with defaults and print
		out = exp.run()
		print(out)
	


if __name__ == "__main__":
	main()

