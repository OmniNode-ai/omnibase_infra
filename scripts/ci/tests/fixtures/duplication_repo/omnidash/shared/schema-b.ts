export const users = pgTable("users", { name: text("name") });
export const comments = pgTable("comments", { id: integer("id") });
