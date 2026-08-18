import { DashboardApp, type ViewKey } from "../DashboardApp";

export default async function DashboardView({ params }: { params: Promise<{ view: string }> }) {
  const { view } = await params;
  return <DashboardApp initialView={view as ViewKey} />;
}
