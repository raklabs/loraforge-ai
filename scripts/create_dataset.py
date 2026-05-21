"""Reproducible scenario-isolated dataset generator for LoraForge AI."""

from __future__ import annotations

import json
from pathlib import Path

# Each line is one scenario with two paraphrases. Split strictly at scenario boundaries:
# 20 train scenarios, 5 validation scenarios, 5 test scenarios per category.
SCENARIOS = {
    "billing": """
My card was charged twice for the monthly plan.|I see two subscription payments for this month instead of one.
Please send me a receipt for yesterday's payment.|Where can I download proof of the payment I made yesterday?
The annual subscription price on my bill is wrong.|My yearly plan invoice lists a different price from checkout.
Can I get a refund for an accidental purchase?|I bought a plan by mistake and want my payment returned.
My discount code was accepted but the invoice has no discount.|The promotion disappeared from the amount I was billed.
Why is sales tax missing from my invoice?|My invoice does not show a tax line; can you correct it?
I cancelled last month but another renewal was charged.|A subscription payment went through after I cancelled.
Can I change the credit card used for future charges?|I need to replace my payment card before the next renewal.
The payment was declined even though my card has funds.|My subscription charge failed despite sufficient card balance.
I need the company address corrected on an invoice.|Please reissue my bill with our updated business address.
Can we pay the subscription by bank transfer?|Is a wire transfer accepted to settle our subscription bill?
I paid for five seats but was invoiced for eight.|The billed seat quantity is higher than the five seats purchased.
A refund was promised but has not reached my bank.|When will the money from my approved refund arrive?
The trial ended and I did not expect a subscription charge.|Why did you take a payment after my free trial?
Please explain the prorated amount after my upgrade.|How was the partial-month upgrade charge calculated?
My invoice currency is euros but I expected pounds.|Could you explain why my bill is in EUR rather than GBP?
Where can I find my outstanding invoices?|I need a list of bills that still need payment.
Our finance team needs a purchase order number on the bill.|Please add our PO reference to the invoice.
I want to switch from monthly to annual billing.|How do I change my subscription payment frequency to yearly?
What date will my next subscription payment be taken?|When does the next charge for my plan fall due?
The same renewal was paid by both transfer and card.|We settled a renewal by bank transfer and were also charged by card.
I was billed for storage I did not purchase.|There is an unexplained storage add-on charge on my invoice.
Can you divide this invoice into two instalments?|Is it possible to pay the bill in two parts?
Our nonprofit exemption was not reflected in the tax charged.|The bill includes tax despite our approved nonprofit exemption.
A late payment fee appeared although I paid on time.|Please investigate a late fee on an invoice settled before its due date.
A cancelled seat still appears on this month's bill.|Why am I paying for a seat removed before the billing period?
Our credit balance was not used to pay the renewal.|You charged my card instead of applying the credit on our billing account.
Please combine the two department invoices into one bill.|Can finance receive a single invoice covering both departments?
The refund amount is smaller than the approved amount.|Only part of the agreed refund was returned to my card.
The introductory rate expired sooner than the offer said.|My bill switched to full price before the promotional period ended.
""",
    "account": """
I forgot my password and cannot sign in.|How can I reset a password I no longer remember?
The password reset email never arrives.|I requested a reset link but my inbox is empty.
My login says the account is locked after repeated attempts.|Too many failed sign-ins have locked me out of my account.
I lost the phone used for two-factor authentication.|How can I regain access without my authentication phone?
Please change the email address on my profile.|I need to update my account's primary email.
I want to replace the display name on my profile.|Where do I edit the name shown on my account?
My verification link has expired.|Can I get a new link to verify my account email?
I would like to permanently delete my account.|How do I close my profile and remove my account?
I cannot remember which email I registered with.|Can you help identify the email associated with my login?
My account was suspended and I need access restored.|How do I appeal a suspension preventing me from signing in?
How can I add a recovery email to my login?|I want to configure a backup email for account recovery.
Someone else changed my account password.|I think an intruder replaced my sign-in credentials.
I need to sign out my account on a lost laptop.|How can I revoke the login session on a missing computer?
My invitation to join the company account has expired.|Can you resend my expired workspace membership invitation?
I accidentally created two profiles and want to merge them.|Can duplicate user accounts be combined into one login?
How do I turn on two-factor authentication for my login?|I want an extra verification step when I sign in.
I changed my phone number and need to update my profile.|Where can I replace the contact phone stored on my account?
I am leaving the company and need to transfer account ownership.|How do I give another administrator ownership of our account?
The login page says no user exists for my email.|My registered address is not recognized when I try to sign in.
I want to change the password while I still know the old one.|Where do I update my current sign-in password?
My recovery codes were lost and I cannot complete login.|I need access restored because I no longer have my backup authentication codes.
An unfamiliar device is listed in my active login sessions.|Can you help remove an unknown device signed into my account?
My organization requires single sign-on; how do I link my login?|I need to connect my existing user account to our company SSO identity.
I no longer receive the SMS code needed to sign in.|Login verification messages are not reaching my phone.
I removed my authenticator by accident and cannot access my profile.|My authentication app entry was deleted and now sign-in is blocked.
My profile is using my old surname and I need it corrected.|How can I update the family name shown in my user profile?
I need to remove a former employee's access to our workspace.|How do I revoke workspace membership for someone who left the company?
I deactivated my account last week and want to reactivate it.|Can my previously disabled user profile be restored?
Our administrator left and nobody can manage user access.|We need help recovering administrator ownership of our organization account.
I accepted an invitation using the wrong personal email.|How do I move my workspace membership to the correct email identity?
""",
    "technical": """
The dashboard displays a server error when I open it.|Opening the dashboard gives me an HTTP 500 error.
Exporting a CSV produces an empty file despite visible rows.|The CSV export has no data even though the table has records.
The mobile app crashes when I attach a photo.|Adding a picture makes the phone application close unexpectedly.
Search returns no results for records I can see in the list.|Existing items cannot be found using the search box.
The chart totals do not match the numbers in the table.|The graph shows different values from its underlying data rows.
The save button does nothing after I edit a document.|My document changes are lost because clicking Save has no effect.
File uploads stop at fifty percent.|Every attachment upload gets stuck halfway through.
The report takes several minutes to load.|Loading a report is much slower than it used to be.
The dark theme makes the toolbar text unreadable.|Toolbar labels blend into the background in dark mode.
Scheduled exports are not running at the configured time.|The automatic export job fails to start on schedule.
The API returns an internal error for a valid request.|A correctly formed API call responds with a server failure.
The calendar shows events one hour later than they should be.|All calendar entries are shifted forward by sixty minutes.
Dragging a task to another column duplicates it.|Moving a card on the board creates an extra copy.
Notifications keep arriving for events I have already read.|Read alerts repeatedly appear as new notifications.
The desktop app freezes when I open a large spreadsheet.|Opening a big sheet causes the desktop client to hang.
The PDF export cuts off the right side of the page.|Exported PDF documents are missing their rightmost columns.
My edits do not sync between the browser and the app.|Changes on the web never appear in the mobile client.
The filter ignores the date range I select.|Results include dates outside the configured filter window.
Audio recordings play with no sound in the built-in player.|The application's recording player is silent for valid audio files.
Clicking a help link inside the app opens a missing page.|An in-app documentation link leads to a 404 page.
The progress bar stays at zero after a batch job completes.|Completed batch tasks still show no progress in the interface.
A webhook sends the same event dozens of times.|Our integration receives repeated copies of a single webhook event.
The map pins appear on the wrong continent.|Location markers are plotted far away from the saved coordinates.
The editor removes line breaks when I paste formatted text.|Pasted paragraphs are flattened into one line by the editor.
Report emails contain broken images.|Images embedded in the automatically emailed reports do not load.
Sorting invoice records by date puts them in alphabetical order.|The invoice list's date-sort feature orders text rather than actual dates.
The print preview is blank for a document with content.|Printing an existing document shows only an empty preview page.
The video player stops whenever I seek backwards.|Rewinding a video causes playback to freeze.
Deleted records reappear after refreshing the page.|Refreshing restores items that the interface said were successfully deleted.
The app consumes all available memory during a data import.|Importing a dataset makes application memory usage grow until it crashes.
""",
    "other": """
Hello there.|Hi, support team.
Can you help me?|I need assistance but have not described the problem yet.
Thank you for your help yesterday.|I appreciate the support you provided recently.
What are your customer service opening hours?|When is the support team available?
Do you offer training sessions for new customers?|Can beginners attend a product onboarding workshop?
Where can I read your privacy policy?|Please share the page explaining your data privacy policy.
Are you hiring software engineers?|Where can I find engineering job openings at your company?
Can our company become a reseller?|I would like information about your reseller partnership program.
Do you have a public product roadmap?|Where can I see your planned future features?
Please add a feature that translates documents into Welsh.|I would like Welsh translation support in a future release.
Is there a community forum for users?|Where can I meet other customers online?
May I interview someone from your company for an article?|I am a journalist requesting an interview with your team.
Where is your company headquarters?|What city is your main office located in?
Can you sponsor our student hackathon?|We are looking for corporate sponsorship for a coding event.
Do you have documentation in Spanish?|Where can I find a Spanish-language user guide?
I would like to arrange a product demonstration.|Can your sales team show our company the product?
How can I submit general feedback?|Where should I send suggestions about your service?
Do you publish a newsletter?|How can I subscribe to company news by email?
I sent this message to the wrong company.|Sorry, this request was intended for another business.
Can I download your brand logo for an event slide?|Where are your approved logos and brand assets?
Will your team attend the education conference this year?|Are you exhibiting at the upcoming education industry event?
Do you provide accessibility conformance documentation?|Where can I obtain a statement about product accessibility standards?
Could you add support for right-to-left document layouts?|Please consider a future feature for right-to-left page design.
What is your company's environmental policy?|Where can I read about your sustainability commitments?
I am a supplier offering office furniture.|Who handles purchasing desks and chairs for your offices?
Can you share a customer case study from the retail sector?|I would like an example of a retailer using your product.
Does your team host webinars about best practices?|Where can I register for a session on effective product use?
I would like to translate your help articles as a volunteer.|Is there a community program for contributing documentation translations?
Please contact me about a university research collaboration.|Our research group wants to explore a partnership with your company.
I have a question but will send the details later.|I will describe what I need in my next message.
""",
}


def build(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "test": []}
    for category, block in SCENARIOS.items():
        scenarios = block.strip().splitlines()
        assert len(scenarios) == 30, (category, len(scenarios))
        for index, line in enumerate(scenarios):
            split = "train" if index < 20 else "validation" if index < 25 else "test"
            variants = line.split("|")
            assert len(variants) == 2
            for variant, text in enumerate(variants):
                scenario = f"{category}-{index + 1:02d}"
                splits[split].append(
                    {
                        "id": f"{scenario}-{variant + 1}",
                        "scenario_id": scenario,
                        "text": text,
                        "category": category,
                    }
                )
    for split, rows in splits.items():
        # Interleave categories so even the smoke subset has varied targets.
        rows.sort(key=lambda row: (row["scenario_id"].rsplit("-", 1)[1], row["id"][-1], row["category"]))
        (destination / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="Use a new directory; preserve frozen experiment data")
    args = p.parse_args()
    if any(args.output.glob("*.jsonl")):
        p.error("Destination already contains JSONL data; choose a new directory")
    build(args.output)
